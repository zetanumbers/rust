//! This pass is only used for the UNIT TESTS and DEBUGGING NEEDS
//! around dependency graph construction. It serves two purposes; it
//! will dump graphs in graphviz form to disk, and it searches for
//! `#[rustc_if_this_changed]` and `#[rustc_then_this_would_need]`
//! annotations. These annotations can be used to test whether paths
//! exist in the graph. These checks run after codegen, so they view the
//! the final state of the dependency graph. Note that there are
//! similar assertions found in `persist::dirty_clean` which check the
//! **initial** state of the dependency graph, just after it has been
//! loaded from disk.
//!
//! In this code, we report errors on each `rustc_if_this_changed`
//! annotation. If a path exists in all cases, then we would report
//! "all path(s) exist". Otherwise, we report: "no path to `foo`" for
//! each case where no path exists. `ui` tests can then be
//! used to check when paths exist or do not.
//!
//! The full form of the `rustc_if_this_changed` annotation is
//! `#[rustc_if_this_changed("foo")]`, which will report a
//! source node of `foo(def_id)`. The `"foo"` is optional and
//! defaults to `"Hir"` if omitted.
//!
//! Example:
//!
//! ```ignore (needs flags)
//! #[rustc_if_this_changed(Hir)]
//! fn foo() { }
//!
//! #[rustc_then_this_would_need(codegen)] //~ ERROR no path from `foo`
//! fn bar() { }
//!
//! #[rustc_then_this_would_need(codegen)] //~ ERROR OK
//! fn baz() { foo(); }
//! ```

use std::cell::Cell;
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};
use std::{cmp, env, ops};

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_data_structures::graph::linked_graph::{Direction, INCOMING, NodeIndex, OUTGOING};
use rustc_hir::def_id::{CRATE_DEF_ID, DefId, LocalDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_middle::dep_graph::{
    DepCache, DepContext, DepGraphQuery, DepKind, DepNode, DepNodeExt, DepNodeFilter, dep_kinds,
};
use rustc_middle::hir::nested_filter;
use rustc_middle::span_bug;
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::tls::with_context;
use rustc_span::{Span, Symbol, sym};
use tracing::debug;
use {rustc_graphviz as dot, rustc_hir as hir};

use crate::errors;

#[allow(missing_docs)]
pub(crate) fn assert_dep_graph(tcx: TyCtxt<'_>) {
    tcx.dep_graph.with_ignore(|| {
        if tcx.sess.opts.unstable_opts.dump_dep_graph {
            tcx.dep_graph.with_query(dump_graph);
        }

        if !tcx.sess.opts.unstable_opts.query_dep_graph {
            return;
        }

        // if the `rustc_attrs` feature is not enabled, then the
        // attributes we are interested in cannot be present anyway, so
        // skip the walk.
        if !tcx.features().rustc_attrs() {
            return;
        }

        // Find annotations supplied by user (if any).
        let (if_this_changed, then_this_would_need) = {
            let mut visitor =
                IfThisChanged { tcx, if_this_changed: vec![], then_this_would_need: vec![] };
            visitor.process_attrs(CRATE_DEF_ID);
            tcx.hir_visit_all_item_likes_in_crate(&mut visitor);
            (visitor.if_this_changed, visitor.then_this_would_need)
        };

        if !if_this_changed.is_empty() || !then_this_would_need.is_empty() {
            assert!(
                tcx.sess.opts.unstable_opts.query_dep_graph,
                "cannot use the `#[{}]` or `#[{}]` annotations \
                    without supplying `-Z query-dep-graph`",
                sym::rustc_if_this_changed,
                sym::rustc_then_this_would_need
            );
        }

        // Check paths.
        check_paths(tcx, &if_this_changed, &then_this_would_need);
    })
}

type Sources = Vec<(Span, DefId, DepNode)>;
type Targets = Vec<(Span, Symbol, hir::HirId, DepNode)>;

struct IfThisChanged<'tcx> {
    tcx: TyCtxt<'tcx>,
    if_this_changed: Sources,
    then_this_would_need: Targets,
}

impl<'tcx> IfThisChanged<'tcx> {
    fn argument(&self, attr: &hir::Attribute) -> Option<Symbol> {
        let mut value = None;
        for list_item in attr.meta_item_list().unwrap_or_default() {
            match list_item.ident() {
                Some(ident) if list_item.is_word() && value.is_none() => value = Some(ident.name),
                _ =>
                // FIXME better-encapsulate meta_item (don't directly access `node`)
                {
                    span_bug!(list_item.span(), "unexpected meta-item {:?}", list_item)
                }
            }
        }
        value
    }

    fn process_attrs(&mut self, def_id: LocalDefId) {
        let def_path_hash = self.tcx.def_path_hash(def_id.to_def_id());
        let hir_id = self.tcx.local_def_id_to_hir_id(def_id);
        let attrs = self.tcx.hir_attrs(hir_id);
        for attr in attrs {
            if attr.has_name(sym::rustc_if_this_changed) {
                let dep_node_interned = self.argument(attr);
                let dep_node = match dep_node_interned {
                    None => DepNode::from_def_path_hash(
                        self.tcx,
                        def_path_hash,
                        dep_kinds::opt_hir_owner_nodes,
                    ),
                    Some(n) => {
                        match DepNode::from_label_string(self.tcx, n.as_str(), def_path_hash) {
                            Ok(n) => n,
                            Err(()) => self.tcx.dcx().emit_fatal(errors::UnrecognizedDepNode {
                                span: attr.span(),
                                name: n,
                            }),
                        }
                    }
                };
                self.if_this_changed.push((attr.span(), def_id.to_def_id(), dep_node));
            } else if attr.has_name(sym::rustc_then_this_would_need) {
                let dep_node_interned = self.argument(attr);
                let dep_node = match dep_node_interned {
                    Some(n) => {
                        match DepNode::from_label_string(self.tcx, n.as_str(), def_path_hash) {
                            Ok(n) => n,
                            Err(()) => self.tcx.dcx().emit_fatal(errors::UnrecognizedDepNode {
                                span: attr.span(),
                                name: n,
                            }),
                        }
                    }
                    None => {
                        self.tcx.dcx().emit_fatal(errors::MissingDepNode { span: attr.span() });
                    }
                };
                self.then_this_would_need.push((
                    attr.span(),
                    dep_node_interned.unwrap(),
                    hir_id,
                    dep_node,
                ));
            }
        }
    }
}

impl<'tcx> Visitor<'tcx> for IfThisChanged<'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        self.process_attrs(item.owner_id.def_id);
        intravisit::walk_item(self, item);
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem<'tcx>) {
        self.process_attrs(trait_item.owner_id.def_id);
        intravisit::walk_trait_item(self, trait_item);
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        self.process_attrs(impl_item.owner_id.def_id);
        intravisit::walk_impl_item(self, impl_item);
    }

    fn visit_field_def(&mut self, s: &'tcx hir::FieldDef<'tcx>) {
        self.process_attrs(s.def_id);
        intravisit::walk_field_def(self, s);
    }
}

fn check_paths<'tcx>(tcx: TyCtxt<'tcx>, if_this_changed: &Sources, then_this_would_need: &Targets) {
    // Return early here so as not to construct the query, which is not cheap.
    if if_this_changed.is_empty() {
        for &(target_span, _, _, _) in then_this_would_need {
            tcx.dcx().emit_err(errors::MissingIfThisChanged { span: target_span });
        }
        return;
    }
    tcx.dep_graph.with_query(|query| {
        for &(_, source_def_id, ref source_dep_node) in if_this_changed {
            let dependents = query.transitive_predecessors(source_dep_node);
            for &(target_span, ref target_pass, _, ref target_dep_node) in then_this_would_need {
                if !dependents.contains(&target_dep_node) {
                    tcx.dcx().emit_err(errors::NoPath {
                        span: target_span,
                        source: tcx.def_path_str(source_def_id),
                        target: *target_pass,
                    });
                } else {
                    tcx.dcx().emit_err(errors::Ok { span: target_span });
                }
            }
        }
    });
}

#[allow(rustc::potential_query_instability)]
fn dump_graph(query: &DepGraphQuery) {
    let path: String = env::var("RUST_DEP_GRAPH").unwrap_or_else(|_| "dep_graph".to_string());

    #[derive(Clone, Copy, PartialEq, Eq)]
    struct ChildIdx(u32);

    impl ChildIdx {
        const DEP_CACHE_BIT: u32 = 1 << 31;

        const fn new(index: u32, cache: DepCache) -> Self {
            debug_assert!(index & Self::DEP_CACHE_BIT == 0);
            ChildIdx(
                index
                    | match cache {
                        DepCache::Cached => 0,
                        DepCache::Computed => Self::DEP_CACHE_BIT,
                    },
            )
        }

        const fn dep_cache(self) -> DepCache {
            if self.0 & Self::DEP_CACHE_BIT != 0 { DepCache::Computed } else { DepCache::Cached }
        }

        const fn index(self) -> u32 {
            self.0 & !Self::DEP_CACHE_BIT
        }
    }

    struct Timeframe {
        dep_kind: DepKind,
        realtime: u32,
        own: u32,
        children: Vec<ChildIdx>,
        compute_tree: Cell<ComputeIdx>,
        realtime_tree: Cell<ComputeIdx>,
    }

    #[derive(Debug, Clone, Copy)]
    struct ComputeNode {
        left: ComputeIdx,
        right: ComputeIdx,
        neighborhood: IndexNeighborhood,
        compute: u32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct IndexNeighborhood(u32);

    impl IndexNeighborhood {
        #[inline(always)]
        fn new(certain_idx: u32) -> Self {
            IndexNeighborhood(certain_idx << 1)
        }

        #[inline(always)]
        fn uncertain_bitmask(self) -> u32 {
            self.0 ^ (self.0 + 1)
        }

        #[inline]
        fn cmp_or_union(self, other: Self) -> Result<cmp::Ordering, Self> {
            let self_unknown_bitmask = self.uncertain_bitmask();
            let other_unknown_bitmask = other.uncertain_bitmask();
            let known_bitmask = !(self_unknown_bitmask | other_unknown_bitmask);
            let distance = (self.0 ^ other.0) & known_bitmask;
            if distance == 0 {
                Ok(self_unknown_bitmask.cmp(&other_unknown_bitmask))
            } else {
                let unknown_bitmask =
                    (distance + 1).checked_next_power_of_two().map(|n| n - 1).unwrap_or(u32::MAX);
                Err(IndexNeighborhood(
                    self.0 & other.0 & !unknown_bitmask | unknown_bitmask.wrapping_shr(1),
                ))
            }
        }
    }

    impl cmp::PartialOrd for IndexNeighborhood {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
            let self_unknown_bitmask = self.uncertain_bitmask();
            let other_unknown_bitmask = other.uncertain_bitmask();
            let known_bitmask = !(self_unknown_bitmask | other_unknown_bitmask);
            ((self.0 ^ other.0) & known_bitmask == 0)
                .then(|| self_unknown_bitmask.cmp(&other_unknown_bitmask))
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ComputeIdx(u32);

    impl ComputeIdx {
        #[inline(always)]
        const fn null() -> ComputeIdx {
            // Has to be top element
            ComputeIdx(u32::MAX)
        }

        #[inline(always)]
        const fn is_null(self) -> bool {
            self.0 == ComputeIdx::null().0
        }

        #[inline(always)]
        const fn is_either_null(self, other: ComputeIdx) -> bool {
            self.is_null() || other.is_null()
        }

        #[inline]
        fn exclusive_or(self, other: ComputeIdx) -> Option<ComputeIdx> {
            self.is_either_null(other).then(|| ComputeIdx(self.0 & other.0))
        }

        #[inline(always)]
        const fn is_both_null(self, other: ComputeIdx) -> bool {
            self.0 & other.0 == ComputeIdx::null().0
        }
    }

    struct ComputeRing {
        data: Vec<ComputeNode>,
        union_mode: bool,
        difference_mode: bool,
    }

    impl ops::Index<ComputeIdx> for ComputeRing {
        type Output = ComputeNode;

        fn index(&self, index: ComputeIdx) -> &Self::Output {
            &self.data[index.0 as usize]
        }
    }

    impl ComputeRing {
        const fn new() -> Self {
            ComputeRing { data: Vec::new(), union_mode: false, difference_mode: false }
        }

        fn alloc_node(&mut self, node: ComputeNode) -> ComputeIdx {
            debug_assert!(
                node.left.is_both_null(node.right) || !node.left.is_either_null(node.right)
            );
            if self.data.try_reserve(1).is_err() {
                self.data = Vec::new();
                panic!("Out of memory!");
            }
            let new_idx = ComputeIdx(u32::try_from(self.data.len()).unwrap());
            self.data.push(node);
            new_idx
        }

        fn alloc_union_node<F>(&mut self, f: F) -> ComputeIdx
        where
            F: FnOnce(&mut Self) -> ComputeNode,
        {
            if self.union_mode {
                let node = f(self);
                self.alloc_node(node)
            } else {
                ComputeIdx::null()
            }
        }

        fn alloc_difference_node<F>(&mut self, f: F) -> ComputeIdx
        where
            F: FnOnce(&mut Self) -> ComputeNode,
        {
            if self.difference_mode {
                let node = f(self);
                self.alloc_node(node)
            } else {
                ComputeIdx::null()
            }
        }

        fn alloc_leaf_node(&mut self, own: u32, timeframe_idx: u32) -> ComputeIdx {
            self.alloc_node(ComputeNode {
                left: ComputeIdx::null(),
                right: ComputeIdx::null(),
                compute: own,
                neighborhood: IndexNeighborhood::new(timeframe_idx),
            })
        }

        fn union(&mut self, lhs: ComputeIdx, rhs: ComputeIdx) -> ComputeIdx {
            self.union_mode = true;
            self.difference_mode = false;
            self.mode_aware_union_and_difference(lhs, rhs)[0]
        }

        fn difference(&mut self, lhs: ComputeIdx, rhs: ComputeIdx) -> ComputeIdx {
            self.union_mode = false;
            self.difference_mode = true;
            self.mode_aware_union_and_difference(lhs, rhs)[1]
        }

        fn union_and_difference(&mut self, lhs: ComputeIdx, rhs: ComputeIdx) -> [ComputeIdx; 2] {
            self.union_mode = true;
            self.difference_mode = true;
            self.mode_aware_union_and_difference(lhs, rhs)
        }

        fn mode_aware_union_and_difference(
            &mut self,
            lhs: ComputeIdx,
            rhs: ComputeIdx,
        ) -> [ComputeIdx; 2] {
            if let Some(union) = lhs.exclusive_or(rhs) {
                return [union, lhs];
            }
            if lhs == rhs {
                return [lhs, ComputeIdx::null()];
            }
            let lhs_node = self[lhs];
            let rhs_node = self[rhs];

            match lhs_node.neighborhood.cmp_or_union(rhs_node.neighborhood) {
                Ok(cmp::Ordering::Equal) => {
                    debug_assert_ne!(
                        lhs_node.neighborhood.0 % 2,
                        0,
                        "Duplicate leaves (lhs: {lhs:?}, rhs: {rhs:?})"
                    );
                    let [union_left, diff_left] =
                        self.mode_aware_union_and_difference(lhs_node.left, rhs_node.left);
                    let [union_right, diff_right] =
                        self.mode_aware_union_and_difference(lhs_node.right, rhs_node.right);
                    let union = if [union_left, union_right] == [lhs_node.left, lhs_node.right] {
                        lhs
                    } else if [union_left, union_right] == [rhs_node.left, rhs_node.right] {
                        rhs
                    } else {
                        self.alloc_union_node(|this| ComputeNode {
                            left: union_left,
                            right: union_right,
                            neighborhood: lhs_node.neighborhood,
                            compute: this[union_left].compute + this[union_right].compute,
                        })
                    };
                    let difference = if diff_left.is_both_null(diff_right) {
                        ComputeIdx::null()
                    } else if [diff_left, diff_right] == [lhs_node.left, lhs_node.right] {
                        lhs
                    } else {
                        diff_left.exclusive_or(diff_right).unwrap_or_else(|| {
                            self.alloc_difference_node(|this| ComputeNode {
                                left: diff_left,
                                right: diff_right,
                                neighborhood: lhs_node.neighborhood,
                                compute: this[diff_left].compute + this[diff_right].compute,
                            })
                        })
                    };
                    [union, difference]
                }
                Err(disjoint_union) => {
                    let [left, right] = if lhs_node.neighborhood.0 < rhs_node.neighborhood.0 {
                        [lhs, rhs]
                    } else {
                        [rhs, lhs]
                    };
                    [
                        self.alloc_union_node(|_| ComputeNode {
                            left,
                            right,
                            neighborhood: disjoint_union,
                            compute: lhs_node.compute + rhs_node.compute,
                        }),
                        lhs,
                    ]
                }
                Ok(cmp::Ordering::Less) => {
                    self.union_and_difference_included(rhs, &rhs_node, lhs, &lhs_node, true)
                }
                Ok(cmp::Ordering::Greater) => {
                    self.union_and_difference_included(lhs, &lhs_node, rhs, &rhs_node, false)
                }
            }
        }

        fn union_and_difference_included(
            &mut self,
            larger: ComputeIdx,
            larger_node: &ComputeNode,
            smaller: ComputeIdx,
            smaller_node: &ComputeNode,
            larger_is_rhs: bool,
        ) -> [ComputeIdx; 2] {
            debug_assert!(smaller_node.neighborhood < larger_node.neighborhood);
            let [union_left, union_right, diff_left, diff_right, diff_new];
            if smaller_node.neighborhood.0 <= larger_node.neighborhood.0 {
                let [lhs, rhs] = if larger_is_rhs {
                    [smaller, larger_node.left]
                } else {
                    [larger_node.left, smaller]
                };
                [union_left, diff_left] = self.mode_aware_union_and_difference(lhs, rhs);
                diff_new = diff_left;
                union_right = larger_node.right;
                diff_right = larger_node.right;
            } else {
                let [lhs, rhs] = if larger_is_rhs {
                    [smaller, larger_node.right]
                } else {
                    [larger_node.right, smaller]
                };
                [union_right, diff_right] = self.mode_aware_union_and_difference(lhs, rhs);
                diff_new = diff_right;
                union_left = larger_node.left;
                diff_left = larger_node.left;
            };
            let union = if [union_left, union_right] == [larger_node.left, larger_node.right] {
                larger
            } else {
                self.alloc_union_node(|this| ComputeNode {
                    left: union_left,
                    right: union_right,
                    neighborhood: larger_node.neighborhood,
                    compute: this[union_left].compute + this[union_right].compute,
                })
            };
            let difference = if larger_is_rhs {
                diff_new
            } else if [diff_left, diff_right] == [larger_node.left, larger_node.right] {
                larger
            } else {
                diff_left.exclusive_or(diff_right).unwrap_or_else(|| {
                    self.alloc_difference_node(|this| ComputeNode {
                        left: diff_left,
                        right: diff_right,
                        neighborhood: larger_node.neighborhood,
                        compute: this[diff_left].compute + this[diff_right].compute,
                    })
                })
            };
            [union, difference]
        }
    }

    const QUANTIZATION_STEP_NS: u64 = 16;

    assert!(query.graph.len_nodes() < u32::MAX.wrapping_shr(1) as usize);
    let mut side_effect_nodes = Vec::new();
    let mut nodes_to_indices =
        FxHashMap::with_capacity_and_hasher(query.graph.len_nodes(), <_>::default());
    let mut hierarchy: Vec<_> = query
        .graph
        .enumerated_nodes()
        .map(|(idx, node)| {
            if node.data.inner.kind == dep_kinds::SideEffect {
                side_effect_nodes.push(idx.node_id() as u32);
            } else {
                let old = nodes_to_indices.insert(node.data.inner, idx.node_id() as u32);
                debug_assert_eq!(old, None);
            }
            // quantize by 16ns
            let realtime =
                u32::try_from(node.data.timeframe.as_nanos() as u64 / QUANTIZATION_STEP_NS)
                    .unwrap();

            Timeframe {
                dep_kind: node.data.inner.kind,
                realtime,
                own: realtime,
                children: Vec::new(),
                compute_tree: Cell::new(ComputeIdx::null()),
                realtime_tree: Cell::new(ComputeIdx::null()),
            }
        })
        .collect();

    with_context(|icx| {
        let tcx = icx.tcx;
        for edge in query.graph.all_edges() {
            let child_idx = edge.target().node_id();
            let parent_idx = edge.source().node_id();
            let child = &hierarchy[child_idx];
            let child_realtime_ns = child.realtime;
            let child_dep_kind = child.dep_kind;
            let parent = &mut hierarchy[parent_idx];
            if !tcx.dep_kind_info(parent.dep_kind).is_anon
                && child_dep_kind != dep_kinds::SideEffect
                && child_dep_kind != dep_kinds::crate_hash
            {
                parent.children.push(ChildIdx::new(child_idx as u32, edge.data));
                if edge.data == DepCache::Computed {
                    parent.own -= child_realtime_ns
                }
            }
        }
    });

    struct CollectedParallelism {
        completed_queries: ComputeIdx,
        computed_queries: ComputeIdx,
        parallelism_difference: u32,
        count: u32,
        sum_own: u32,
    }

    impl Default for CollectedParallelism {
        fn default() -> Self {
            CollectedParallelism {
                completed_queries: ComputeIdx::null(),
                computed_queries: ComputeIdx::null(),
                parallelism_difference: 0,
                count: 0,
                sum_own: 0,
            }
        }
    }

    let mut ring = ComputeRing::new();
    let mut compute_parallelism = FxHashMap::<DepKind, CollectedParallelism>::default();

    let mut update = Instant::now();
    let mut unclaimed_interspersed_kinds = FxHashSet::default();
    let mut unclaimed_nodes = Vec::new();
    for i in 0..hierarchy.len() as u32 {
        let timeframe = &hierarchy[i as usize];

        let mut unclaimed_iter = unclaimed_nodes.iter().rev();
        for child in timeframe.children.iter().rev() {
            if child.dep_cache() == DepCache::Cached {
                continue;
            }
            loop {
                let unclaimed = *unclaimed_iter.next().unwrap();
                if unclaimed == child.index() {
                    break;
                }
                unclaimed_interspersed_kinds.insert(hierarchy[unclaimed as usize].dep_kind);
            }
        }
        unclaimed_nodes.resize_with(unclaimed_iter.len(), || panic!());
        if timeframe.dep_kind != dep_kinds::SideEffect {
            unclaimed_nodes.push(i);
        }

        let this_leaf = ring.alloc_leaf_node(timeframe.own, i);
        let (this_total_tree, this_real_tree) = timeframe.children.iter().fold(
            (this_leaf, this_leaf),
            |(acc_total, acc_real), &child| {
                let child_node = &hierarchy[child.index() as usize];
                let child_total_tree = child_node.compute_tree.get();
                debug_assert!(!child_total_tree.is_null());
                let new_total_tree = ring.union(acc_total, child_total_tree);

                let new_real_tree = match child.dep_cache() {
                    DepCache::Computed => {
                        let child_real_tree = child_node.realtime_tree.get();
                        debug_assert!(!child_real_tree.is_null());
                        ring.union(acc_real, child_real_tree)
                    }
                    DepCache::Cached => acc_real,
                };
                (new_total_tree, new_real_tree)
            },
        );
        timeframe.compute_tree.set(this_total_tree);
        timeframe.realtime_tree.set(this_real_tree);

        let entry = compute_parallelism.entry(timeframe.dep_kind).or_default();
        let [new_completed_queries, new_tree] =
            ring.union_and_difference(this_total_tree, entry.completed_queries);
        let new_computed_queries = ring.union(this_real_tree, entry.computed_queries);

        let compute_sum = ring[new_tree].compute;
        let timeframe = &hierarchy[i as usize];
        let compute_max = timeframe.own
            + timeframe
                .children
                .iter()
                .map(|&child| {
                    let difference = ring.difference(
                        hierarchy[child.index() as usize].compute_tree.get(),
                        entry.completed_queries,
                    );
                    if difference.is_null() { 0 } else { ring[difference].compute }
                })
                .max()
                .unwrap_or(0);
        entry.parallelism_difference += compute_sum - compute_max;
        entry.completed_queries = new_completed_queries;
        entry.computed_queries = new_computed_queries;
        entry.count += 1;
        entry.sum_own += timeframe.own;
        if i % 256 == 0 {
            let new_update = Instant::now();
            if new_update.duration_since(update) >= Duration::from_secs(1) {
                update = new_update;
                eprintln!("{i}/{}", hierarchy.len());
            }
        }
    }
    eprintln!("unclaimed_interspersed_kinds = {unclaimed_interspersed_kinds:#?}");

    let mut compute_kinds: Vec<_> = compute_parallelism.into_iter().collect();
    compute_kinds.sort_unstable_by_key(|(_, p)| p.parallelism_difference);

    {
        // dump a .txt file with just the edges:
        let txt_path = format!("{path}.csv");
        let mut file = File::create_buffered(&txt_path).unwrap();
        writeln!(file, "query,count,parallelism_ns,sum_realtime_ns,sum_own_ns").unwrap();
        for (kind, data) in compute_kinds.into_iter().rev() {
            let compute = (data.parallelism_difference as u64) * QUANTIZATION_STEP_NS;
            let count = data.count;
            let sum_realtime = (ring[data.computed_queries].compute as u64) * QUANTIZATION_STEP_NS;
            let sum_own = (data.sum_own as u64) * QUANTIZATION_STEP_NS;
            writeln!(file, "{kind:?},{count},{compute},{sum_realtime},{sum_own}").unwrap();
        }
    }
}

#[allow(missing_docs)]
struct GraphvizDepGraph(FxIndexSet<DepKind>, Vec<(DepKind, DepKind)>);

impl<'a> dot::GraphWalk<'a> for GraphvizDepGraph {
    type Node = DepKind;
    type Edge = (DepKind, DepKind);
    fn nodes(&self) -> dot::Nodes<'_, DepKind> {
        let nodes: Vec<_> = self.0.iter().cloned().collect();
        nodes.into()
    }
    fn edges(&self) -> dot::Edges<'_, (DepKind, DepKind)> {
        self.1[..].into()
    }
    fn source(&self, edge: &(DepKind, DepKind)) -> DepKind {
        edge.0
    }
    fn target(&self, edge: &(DepKind, DepKind)) -> DepKind {
        edge.1
    }
}

impl<'a> dot::Labeller<'a> for GraphvizDepGraph {
    type Node = DepKind;
    type Edge = (DepKind, DepKind);
    fn graph_id(&self) -> dot::Id<'_> {
        dot::Id::new("DependencyGraph").unwrap()
    }
    fn node_id(&self, n: &DepKind) -> dot::Id<'_> {
        let s: String = format!("{n:?}")
            .chars()
            .map(|c| if c == '_' || c.is_alphanumeric() { c } else { '_' })
            .collect();
        debug!("n={:?} s={:?}", n, s);
        dot::Id::new(s).unwrap()
    }
    fn node_label(&self, n: &DepKind) -> dot::LabelText<'_> {
        dot::LabelText::label(format!("{n:?}"))
    }
}

// Given an optional filter like `"x,y,z"`, returns either `None` (no
// filter) or the set of nodes whose labels contain all of those
// substrings.
fn node_set<'q>(
    query: &'q DepGraphQuery,
    filter: &DepNodeFilter,
) -> Option<FxIndexSet<&'q DepNode>> {
    debug!("node_set(filter={:?})", filter);

    if filter.accepts_all() {
        return None;
    }

    Some(query.nodes().into_iter().filter(|n| filter.test(n)).collect())
}

fn filter_nodes<'q>(
    query: &'q DepGraphQuery,
    sources: &Option<FxIndexSet<&'q DepNode>>,
    targets: &Option<FxIndexSet<&'q DepNode>>,
) -> FxIndexSet<DepKind> {
    if let Some(sources) = sources {
        if let Some(targets) = targets {
            walk_between(query, sources, targets)
        } else {
            walk_nodes(query, sources, OUTGOING)
        }
    } else if let Some(targets) = targets {
        walk_nodes(query, targets, INCOMING)
    } else {
        query.nodes().into_iter().map(|n| n.kind).collect()
    }
}

fn walk_nodes<'q>(
    query: &'q DepGraphQuery,
    starts: &FxIndexSet<&'q DepNode>,
    direction: Direction,
) -> FxIndexSet<DepKind> {
    let mut set = FxIndexSet::default();
    for &start in starts {
        debug!("walk_nodes: start={:?} outgoing?={:?}", start, direction == OUTGOING);
        if set.insert(start.kind) {
            let mut stack = vec![query.indices[start]];
            while let Some(index) = stack.pop() {
                for (_, edge) in query.graph.adjacent_edges(index, direction) {
                    let neighbor_index = edge.source_or_target(direction);
                    let neighbor = query.graph.node_data(neighbor_index);
                    if set.insert(neighbor.inner.kind) {
                        stack.push(neighbor_index);
                    }
                }
            }
        }
    }
    set
}

fn walk_between<'q>(
    query: &'q DepGraphQuery,
    sources: &FxIndexSet<&'q DepNode>,
    targets: &FxIndexSet<&'q DepNode>,
) -> FxIndexSet<DepKind> {
    // This is a bit tricky. We want to include a node only if it is:
    // (a) reachable from a source and (b) will reach a target. And we
    // have to be careful about cycles etc. Luckily efficiency is not
    // a big concern!

    #[derive(Copy, Clone, PartialEq)]
    enum State {
        Undecided,
        Deciding,
        Included,
        Excluded,
    }

    let mut node_states = vec![State::Undecided; query.graph.len_nodes()];

    for &target in targets {
        node_states[query.indices[target].0] = State::Included;
    }

    for source in sources.iter().map(|&n| query.indices[n]) {
        recurse(query, &mut node_states, source);
    }

    return query
        .nodes()
        .into_iter()
        .filter(|&n| {
            let index = query.indices[n];
            node_states[index.0] == State::Included
        })
        .map(|n| n.kind)
        .collect();

    fn recurse(query: &DepGraphQuery, node_states: &mut [State], node: NodeIndex) -> bool {
        match node_states[node.0] {
            // known to reach a target
            State::Included => return true,

            // known not to reach a target
            State::Excluded => return false,

            // backedge, not yet known, say false
            State::Deciding => return false,

            State::Undecided => {}
        }

        node_states[node.0] = State::Deciding;

        for neighbor_index in query.graph.successor_nodes(node) {
            if recurse(query, node_states, neighbor_index) {
                node_states[node.0] = State::Included;
            }
        }

        // if we didn't find a path to target, then set to excluded
        if node_states[node.0] == State::Deciding {
            node_states[node.0] = State::Excluded;
            false
        } else {
            assert!(node_states[node.0] == State::Included);
            true
        }
    }
}
