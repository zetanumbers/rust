use std::collections::BTreeMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::io::Write;
use std::num::NonZero;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexmap::{self, IndexMap};
use rustc_data_structures::tree_node_index::TreeNodeIndex;
use rustc_errors::{Diag, DiagCtxtHandle};
use rustc_hir::def::DefKind;
use rustc_session::Session;
use rustc_span::Span;

use super::{QueryStackDeferred, QueryStackFrameExtra};
use crate::dep_graph::DepContext;
use crate::error::CycleStack;
use crate::query::plumbing::CycleError;
use crate::query::{QueryContext, QueryStackFrame};

/// Represents a span and a query key.
#[derive(Clone, Debug)]
pub struct QueryInfo<I> {
    /// The span corresponding to the reason for which this query was required.
    pub span: Span,
    pub frame: QueryStackFrame<I>,
}

impl<'tcx> QueryInfo<QueryStackDeferred<'tcx>> {
    pub(crate) fn lift(&self) -> QueryInfo<QueryStackFrameExtra> {
        QueryInfo { span: self.span, frame: self.frame.lift() }
    }
}

/// Map from query job IDs to job information collected by
/// [`QueryContext::collect_active_jobs_from_all_queries`].
pub type QueryMap<'tcx> = FxHashMap<QueryJobId, QueryJobInfo<'tcx>>;

/// A value uniquely identifying an active query job.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct QueryJobId(pub NonZero<u64>);

impl QueryJobId {
    fn frame<'a, 'tcx>(self, map: &'a QueryMap<'tcx>) -> QueryStackFrame<QueryStackDeferred<'tcx>> {
        map.get(&self).unwrap().frame.clone()
    }
}

#[derive(Clone, Debug)]
pub struct QueryJobInfo<'tcx> {
    pub frame: QueryStackFrame<QueryStackDeferred<'tcx>>,
    pub job: QueryJob<'tcx>,
}

/// Represents an active query job.
#[derive(Debug)]
pub struct QueryJob<'tcx> {
    pub id: QueryJobId,

    /// The span corresponding to the reason for which this query was required.
    pub span: Span,

    /// The parent query job which created this job and is implicitly waiting on it.
    pub parent: Option<QueryInclusion>,

    /// The latch that is used to wait on this job.
    latch: Option<QueryLatch<'tcx>>,
}

impl<'tcx> Clone for QueryJob<'tcx> {
    fn clone(&self) -> Self {
        Self { id: self.id, span: self.span, parent: self.parent, latch: self.latch.clone() }
    }
}

impl<'tcx> QueryJob<'tcx> {
    /// Creates a new query job.
    #[inline]
    pub fn new(id: QueryJobId, span: Span, parent: Option<QueryInclusion>) -> Self {
        QueryJob { id, span, parent, latch: None }
    }

    pub(super) fn latch(&mut self) -> QueryLatch<'tcx> {
        if self.latch.is_none() {
            self.latch = Some(QueryLatch::new());
        }
        self.latch.as_ref().unwrap().clone()
    }

    /// Signals to waiters that the query is complete.
    ///
    /// This does nothing for single threaded rustc,
    /// as there are no concurrent jobs which could be waiting on us
    #[inline]
    pub fn signal_complete(self) {
        if let Some(latch) = self.latch {
            latch.set();
        }
    }
}

impl QueryJobId {
    pub(super) fn find_cycle_in_stack<'tcx>(
        &self,
        query_map: QueryMap<'tcx>,
        mut current_job: Option<QueryJobId>,
        span: Span,
    ) -> CycleError<QueryStackDeferred<'tcx>> {
        // Find the waitee amongst `current_job` parents
        let mut cycle = Vec::new();

        while let Some(job) = current_job {
            let info = query_map.get(&job).unwrap();
            cycle.push(QueryInfo { span: info.job.span, frame: info.frame.clone() });

            if job == *self {
                cycle.reverse();

                // This is the end of the cycle
                // The span entry we included was for the usage
                // of the cycle itself, and not part of the cycle
                // Replace it with the span which caused the cycle to form
                cycle[0].span = span;
                // Find out why the cycle itself was used
                let usage = info
                    .job
                    .parent
                    .as_ref()
                    .map(|parent| (info.job.span, parent.id.frame(&query_map)));
                return CycleError { usage, cycle };
            }

            current_job = info.job.parent.map(|i| i.id);
        }

        panic!("did not find a cycle")
    }

    #[cold]
    #[inline(never)]
    pub fn find_dep_kind_root<'tcx>(
        &self,
        query_map: QueryMap<'tcx>,
    ) -> (QueryJobInfo<'tcx>, usize) {
        let mut depth = 1;
        let info = query_map.get(&self).unwrap();
        let dep_kind = info.frame.dep_kind;
        let mut current = info.job.parent;
        let mut last_layout = (info.clone(), depth);

        while let Some(inclusion) = current {
            let info = query_map.get(&inclusion.id).unwrap();
            if info.frame.dep_kind == dep_kind {
                depth += 1;
                last_layout = (info.clone(), depth);
            }
            current = info.job.parent;
        }
        last_layout
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QueryInclusion {
    pub id: QueryJobId,
    pub branch: TreeNodeIndex,
}

#[derive(Debug)]
struct QueryWaiter<'tcx> {
    query: Option<QueryInclusion>,
    condvar: Condvar,
    cycle: Mutex<Option<CycleError<QueryStackDeferred<'tcx>>>>,
}

#[derive(Debug)]
struct QueryLatchInfo<'tcx> {
    complete: bool,
    waiters: Vec<Arc<QueryWaiter<'tcx>>>,
}

#[derive(Debug)]
pub(super) struct QueryLatch<'tcx> {
    info: Arc<Mutex<QueryLatchInfo<'tcx>>>,
}

impl<'tcx> Clone for QueryLatch<'tcx> {
    fn clone(&self) -> Self {
        Self { info: Arc::clone(&self.info) }
    }
}

impl<'tcx> QueryLatch<'tcx> {
    fn new() -> Self {
        QueryLatch {
            info: Arc::new(Mutex::new(QueryLatchInfo { complete: false, waiters: Vec::new() })),
        }
    }

    /// Awaits for the query job to complete.
    pub(super) fn wait_on(
        &self,
        qcx: impl QueryContext<'tcx>,
        query: Option<QueryInclusion>,
    ) -> Result<(), CycleError<QueryStackDeferred<'tcx>>> {
        let waiter =
            Arc::new(QueryWaiter { query, cycle: Mutex::new(None), condvar: Condvar::new() });
        self.wait_on_inner(qcx, &waiter);
        // FIXME: Get rid of this lock. We have ownership of the QueryWaiter
        // although another thread may still have a Arc reference so we cannot
        // use Arc::get_mut
        let mut cycle = waiter.cycle.lock();
        match cycle.take() {
            None => Ok(()),
            Some(cycle) => Err(cycle),
        }
    }

    /// Awaits the caller on this latch by blocking the current thread.
    fn wait_on_inner(&self, qcx: impl QueryContext<'tcx>, waiter: &Arc<QueryWaiter<'tcx>>) {
        let mut info = self.info.lock();
        if !info.complete {
            // We push the waiter on to the `waiters` list. It can be accessed inside
            // the `wait` call below, by 1) the `set` method or 2) by deadlock detection.
            // Both of these will remove it from the `waiters` list before resuming
            // this thread.
            info.waiters.push(Arc::clone(waiter));

            // If this detects a deadlock and the deadlock handler wants to resume this thread
            // we have to be in the `wait` call. This is ensured by the deadlock handler
            // getting the self.info lock.
            rustc_thread_pool::mark_blocked();
            let proxy = qcx.jobserver_proxy();
            proxy.release_thread();
            waiter.condvar.wait(&mut info);
            // Release the lock before we potentially block in `acquire_thread`
            drop(info);
            proxy.acquire_thread();
        }
    }

    /// Sets the latch and resumes all waiters on it
    fn set(&self) {
        let mut info = self.info.lock();
        debug_assert!(!info.complete);
        info.complete = true;
        let registry = rustc_thread_pool::Registry::current();
        for waiter in info.waiters.drain(..) {
            rustc_thread_pool::mark_unblocked(&registry);
            waiter.condvar.notify_one();
        }
    }
}

#[allow(rustc::potential_query_instability)]
pub fn break_query_cycles<'tcx>(query_map: QueryMap<'tcx>, registry: &rustc_thread_pool::Registry) {
    let mut root_query = None;
    for (&query, info) in &query_map {
        if info.job.parent.is_none() {
            assert!(root_query.is_none(), "found multiple threads without start");
            root_query = Some(query);
        }
    }
    let root_query = root_query.expect("no root query was found");

    let mut subqueries = FxHashMap::<_, BTreeMap<TreeNodeIndex, _>>::default();
    for query in query_map.values() {
        let Some(inclusion) = &query.job.parent else {
            continue;
        };
        let old = subqueries
            .entry(inclusion.id)
            .or_default()
            .insert(inclusion.branch, (query.job.id, usize::MAX));
        assert!(old.is_none());
    }

    for query in query_map.values() {
        let Some(latch) = &query.job.latch else {
            continue;
        };
        // Latch mutexes should be at least about to unlock as we do not hold it anywhere too long
        let lock = latch.info.lock();
        assert!(!lock.complete);
        for (waiter_idx, waiter) in lock.waiters.iter().enumerate() {
            let inclusion = waiter.query.expect("cannot wait on a root query");
            let old = subqueries
                .entry(inclusion.id)
                .or_default()
                .insert(inclusion.branch, (query.job.id, waiter_idx));
            assert!(old.is_none());
        }
    }

    let mut visited = IndexMap::new();
    let mut last_usage = None;
    let mut last_waiter_idx = usize::MAX;
    let mut current = root_query;
    while let indexmap::map::Entry::Vacant(entry) = visited.entry(current) {
        entry.insert((last_usage, last_waiter_idx));
        last_usage = Some(current);
        (current, last_waiter_idx) = *subqueries
            .get(&current)
            .unwrap_or_else(|| {
                panic!(
                    "deadlock detected as we're unable to find a query cycle to break\n\
                current query map:\n{:#?}",
                    query_map
                )
            })
            .first_key_value()
            .unwrap()
            .1;
    }
    let usage = visited[&current].0;
    let mut iter = visited.keys().rev();
    let mut cycle = Vec::new();
    loop {
        let query_id = *iter.next().unwrap();
        let query = &query_map[&query_id];
        cycle.push(QueryInfo { span: query.job.span, frame: query.frame.clone() });
        if query_id == current {
            break;
        }
    }

    cycle.reverse();
    let cycle_error = CycleError {
        usage: usage.map(|id| {
            let query = &query_map[&id];
            (query.job.span, query.frame.clone())
        }),
        cycle,
    };

    let (waited_on, waiter_idx) = if last_waiter_idx != usize::MAX {
        (current, last_waiter_idx)
    } else {
        let (&waited_on, &(_, waiter_idx)) =
            visited.iter().rev().find(|(_, (_, waiter_idx))| *waiter_idx != usize::MAX).unwrap();
        (waited_on, waiter_idx)
    };
    let waited_on = &query_map[&waited_on];
    let latch = waited_on.job.latch.as_ref().unwrap();
    let mut latch_info_lock = latch.info.try_lock().unwrap();
    let waiter = latch_info_lock.waiters.remove(waiter_idx);
    let mut cycle_lock = waiter.cycle.try_lock().unwrap();
    assert!(cycle_lock.is_none());
    *cycle_lock = Some(cycle_error);
    rustc_thread_pool::mark_unblocked(registry);
    waiter.condvar.notify_one();
}

#[inline(never)]
#[cold]
pub fn report_cycle<'a>(
    sess: &'a Session,
    CycleError { usage, cycle: stack }: &CycleError,
) -> Diag<'a> {
    assert!(!stack.is_empty());

    let span = stack[0].frame.info.default_span(stack[1 % stack.len()].span);

    let mut cycle_stack = Vec::new();

    use crate::error::StackCount;
    let stack_count = if stack.len() == 1 { StackCount::Single } else { StackCount::Multiple };

    for i in 1..stack.len() {
        let frame = &stack[i].frame;
        let span = frame.info.default_span(stack[(i + 1) % stack.len()].span);
        cycle_stack.push(CycleStack { span, desc: frame.info.description.to_owned() });
    }

    let mut cycle_usage = None;
    if let Some((span, ref query)) = *usage {
        cycle_usage = Some(crate::error::CycleUsage {
            span: query.info.default_span(span),
            usage: query.info.description.to_string(),
        });
    }

    let alias =
        if stack.iter().all(|entry| matches!(entry.frame.info.def_kind, Some(DefKind::TyAlias))) {
            Some(crate::error::Alias::Ty)
        } else if stack.iter().all(|entry| entry.frame.info.def_kind == Some(DefKind::TraitAlias)) {
            Some(crate::error::Alias::Trait)
        } else {
            None
        };

    let cycle_diag = crate::error::Cycle {
        span,
        cycle_stack,
        stack_bottom: stack[0].frame.info.description.to_owned(),
        alias,
        cycle_usage,
        stack_count,
        note_span: (),
    };

    sess.dcx().create_err(cycle_diag)
}

pub fn print_query_stack<'tcx, Qcx: QueryContext<'tcx>>(
    qcx: Qcx,
    mut current_query: Option<QueryJobId>,
    dcx: DiagCtxtHandle<'_>,
    limit_frames: Option<usize>,
    mut file: Option<std::fs::File>,
) -> usize {
    // Be careful relying on global state here: this code is called from
    // a panic hook, which means that the global `DiagCtxt` may be in a weird
    // state if it was responsible for triggering the panic.
    let mut count_printed = 0;
    let mut count_total = 0;

    // Make use of a partial query map if we fail to take locks collecting active queries.
    let query_map = match qcx.collect_active_jobs_from_all_queries(false) {
        Ok(query_map) => query_map,
        Err(query_map) => query_map,
    };

    if let Some(ref mut file) = file {
        let _ = writeln!(file, "\n\nquery stack during panic:");
    }
    while let Some(query) = current_query {
        let Some(query_info) = query_map.get(&query) else {
            break;
        };
        let query_extra = query_info.frame.info.extract();
        if Some(count_printed) < limit_frames || limit_frames.is_none() {
            // Only print to stderr as many stack frames as `num_frames` when present.
            dcx.struct_failure_note(format!(
                "#{} [{:?}] {}",
                count_printed, query_info.frame.dep_kind, query_extra.description
            ))
            .with_span(query_info.job.span)
            .emit();
            count_printed += 1;
        }

        if let Some(ref mut file) = file {
            let _ = writeln!(
                file,
                "#{} [{}] {}",
                count_total,
                qcx.dep_context().dep_kind_vtable(query_info.frame.dep_kind).name,
                query_extra.description
            );
        }

        current_query = query_info.job.parent.map(|i| i.id);
        count_total += 1;
    }

    if let Some(ref mut file) = file {
        let _ = writeln!(file, "end of query stack");
    }
    count_total
}
