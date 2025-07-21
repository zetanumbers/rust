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
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::{cmp, env, mem, ops};

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

    struct Timeframe {
        dep_kind: DepKind,
        realtime_ns: u64,
        own_ns: u64,
        children: FxHashSet<u32>,
        parents: FxHashSet<u32>,
        depth: Cell<u32>,
        compute_tree: Cell<ComputeIdx>,
    }

    #[derive(Clone, Copy)]
    struct ComputeNode {
        left: ComputeIdx,
        right: ComputeIdx,
        shared_high_bits: u32,
        shared_high_bitmask: u32,
        compute_ns: u64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ComputeIdx(usize);

    impl ComputeIdx {
        #[inline]
        const fn null() -> ComputeIdx {
            // Has to be top element
            ComputeIdx(usize::MAX)
        }

        #[inline]
        const fn is_null(self) -> bool {
            self.0 == ComputeIdx::null().0
        }

        #[inline]
        const fn is_either_null(self, other: ComputeIdx) -> bool {
            self.0 | other.0 == ComputeIdx::null().0
        }

        #[inline]
        const fn is_both_null(self, other: ComputeIdx) -> bool {
            self.0 & other.0 == ComputeIdx::null().0
        }
    }

    struct ComputeRing {
        data: Vec<ComputeNode>,
    }

    impl ops::Index<ComputeIdx> for ComputeRing {
        type Output = ComputeNode;

        fn index(&self, index: ComputeIdx) -> &Self::Output {
            &self.data[index.0]
        }
    }

    impl ComputeRing {
        const fn new() -> Self {
            ComputeRing { data: Vec::new() }
        }

        fn len(&self) -> usize {
            self.data.len()
        }

        fn alloc_node(&mut self, node: ComputeNode) -> ComputeIdx {
            let new_idx = self.data.len();
            self.data.push(node);
            ComputeIdx(new_idx)
        }

        fn alloc_leaf_node(&mut self, own_ns: u64, timeframe_idx: u32) -> ComputeIdx {
            self.alloc_node(ComputeNode {
                left: ComputeIdx::null(),
                right: ComputeIdx::null(),
                compute_ns: own_ns,
                shared_high_bitmask: !0,
                shared_high_bits: timeframe_idx,
            })
        }

        fn union(&mut self, mut lhs: ComputeIdx, mut rhs: ComputeIdx) -> ComputeIdx {
            if lhs.is_either_null(rhs) {
                return ComputeIdx(lhs.0 & rhs.0);
            }
            if lhs == rhs {
                return lhs;
            }
            let mut lhs_node = self[lhs];
            let mut rhs_node = self[rhs];
            let xor_high_bits = lhs_node.shared_high_bits ^ rhs_node.shared_high_bits;
            let shared_high_bitmask = !((!0_u32).wrapping_shr(xor_high_bits.leading_zeros()));
            let [left, right] = if lhs_node.shared_high_bits <= rhs_node.shared_high_bits {
                [lhs, rhs]
            } else {
                [rhs, lhs]
            };

            match lhs_node.shared_high_bitmask.cmp(&rhs_node.shared_high_bitmask) {
                cmp::Ordering::Equal => {
                    if xor_high_bits == 0 {
                        let left = self.union(lhs_node.left, rhs_node.left);
                        let right = self.union(lhs_node.right, rhs_node.right);
                        if [left, right] == [lhs_node.left, lhs_node.right] {
                            return lhs;
                        }

                        debug_assert!(!left.is_either_null(right) || left.is_both_null(right));

                        return self.alloc_node(ComputeNode {
                            left,
                            right,
                            shared_high_bits: lhs_node.shared_high_bits,
                            shared_high_bitmask: lhs_node.shared_high_bitmask,
                            compute_ns: self[left].compute_ns + self[right].compute_ns,
                        });
                    } else {
                        return self.alloc_node(ComputeNode {
                            left,
                            right,
                            shared_high_bits: lhs_node.shared_high_bits & shared_high_bitmask,
                            shared_high_bitmask,
                            compute_ns: lhs_node.compute_ns + rhs_node.compute_ns,
                        });
                    }
                }
                cmp::Ordering::Greater => {
                    mem::swap(&mut lhs, &mut rhs);
                    mem::swap(&mut lhs_node, &mut rhs_node);
                }
                cmp::Ordering::Less => (),
            }

            // `lhs` has lesser bitmask

            if xor_high_bits & rhs_node.shared_high_bitmask != rhs_node.shared_high_bits {
                self.alloc_node(ComputeNode {
                    left,
                    right,
                    shared_high_bits: lhs_node.shared_high_bits & shared_high_bitmask,
                    shared_high_bitmask,
                    compute_ns: lhs_node.compute_ns + rhs_node.compute_ns,
                })
            } else {
                let [left, right] = if lhs_node.shared_high_bits
                    <= rhs_node.shared_high_bits | (!rhs_node.shared_high_bitmask).wrapping_shr(1)
                {
                    let left = self.union(lhs_node.left, rhs);
                    if left == lhs_node.left {
                        return lhs;
                    }
                    [left, lhs_node.right]
                } else {
                    let right = self.union(lhs_node.right, rhs);
                    if right == lhs_node.right {
                        return rhs;
                    }
                    [lhs_node.left, right]
                };

                debug_assert!(!left.is_either_null(right) || left.is_both_null(right));

                let get_compute_ns = |idx: ComputeIdx| {
                    if idx.is_null() { 0 } else { self[idx].compute_ns }
                };
                self.alloc_node(ComputeNode {
                    left,
                    right,
                    shared_high_bits: lhs_node.shared_high_bits,
                    shared_high_bitmask: lhs_node.shared_high_bitmask,
                    compute_ns: get_compute_ns(left) + get_compute_ns(right),
                })
            }
        }
    }

    assert!(query.graph.len_nodes() < u32::MAX as usize);
    let mut side_effect_node_count = 0;
    let (mut nodes_to_indices, mut hierarchy): (FxHashMap<_, u32>, Vec<_>) = query
        .graph
        .enumerated_nodes()
        .map(|(idx, node)| {
            if node.data.inner.kind == dep_kinds::SideEffect {
                side_effect_node_count += 1;
            }
            let realtime = if node.data.inner.kind != dep_kinds::crate_hash {
                node.data.timeframe.as_nanos() as u64
            } else {
                0
            };
            (
                (node.data.inner, idx.node_id() as u32),
                Timeframe {
                    dep_kind: node.data.inner.kind,
                    realtime_ns: realtime,
                    own_ns: realtime,
                    children: FxHashSet::default(),
                    parents: FxHashSet::default(),
                    depth: Cell::new(0),
                    compute_tree: Cell::new(ComputeIdx::null()),
                },
            )
        })
        .collect();
    debug_assert_eq!(nodes_to_indices.len() + side_effect_node_count - 1, hierarchy.len());

    with_context(|icx| {
        let tcx = icx.tcx;
        for edge in query.graph.all_edges() {
            let child_idx = edge.target().node_id();
            let parent_idx = edge.source().node_id();
            let child = &mut hierarchy[child_idx];
            let child_realtime_ns = child.realtime_ns;
            let substract_ns = match edge.data {
                DepCache::Computed => child_realtime_ns,
                DepCache::Cached => 0,
            };
            if !child.parents.insert(parent_idx as u32) {
                continue;
            }
            let parent = &mut hierarchy[parent_idx];
            let new_child = parent.children.insert(child_idx as u32);
            debug_assert!(new_child);
            if let Some(new_own) = parent.own_ns.checked_sub(substract_ns) {
                parent.own_ns = new_own;
            } else {
                assert!(tcx.dep_kind_info(parent.dep_kind).is_anon)
            }
        }
    });

    for timeframe in &hierarchy {
        fn calc_depth(timeframe: &Timeframe, hierarchy: &Vec<Timeframe>) -> u32 {
            let depth = timeframe.depth.get();
            if depth != 0 {
                return depth;
            }
            let depth = timeframe
                .parents
                .iter()
                .map(|&i| calc_depth(&hierarchy[i as usize], hierarchy))
                .max()
                .unwrap_or(0)
                + 1;
            timeframe.depth.set(depth);
            depth
        }

        calc_depth(timeframe, &hierarchy);
    }

    {
        let mut backward_permutation: Vec<u32> = (0..hierarchy.len() as u32).collect();
        backward_permutation
            .sort_unstable_by_key(|&i| cmp::Reverse(*hierarchy[i as usize].depth.get_mut()));
        let mut forward_permutation = vec![0_u32; hierarchy.len()];
        for (new, &old) in backward_permutation.iter().enumerate() {
            forward_permutation[old as usize] = new as u32;
        }
        hierarchy = backward_permutation
            .iter()
            .map(|&i| {
                let old = &hierarchy[i as usize];
                let permute = |&i| forward_permutation[i as usize];
                Timeframe {
                    dep_kind: old.dep_kind,
                    realtime_ns: old.realtime_ns,
                    own_ns: old.own_ns,
                    children: old.children.iter().map(permute).collect(),
                    parents: old.parents.iter().map(permute).collect(),
                    depth: old.depth.clone(),
                    compute_tree: old.compute_tree.clone(),
                }
            })
            .collect();

        for idx in nodes_to_indices.values_mut() {
            *idx = forward_permutation[*idx as usize];
        }
    }

    let mut ring = ComputeRing::new();
    let mut compute_parallelism = FxHashMap::<DepKind, u64>::default();

    for i in 0..hierarchy.len() as u32 {
        let timeframe = &hierarchy[i as usize];

        let this_leaf = ring.alloc_leaf_node(timeframe.own_ns, i);
        let this_tree = timeframe.children.iter().fold(this_leaf, |acc, &child| {
            let child_tree = hierarchy[child as usize].compute_tree.get();
            debug_assert_ne!(child_tree, ComputeIdx::null());
            ring.union(acc, child_tree)
        });
        timeframe.compute_tree.set(this_tree);

        let compute_sum_ns = ring[this_tree].compute_ns;
        let timeframe = &hierarchy[i as usize];
        let compute_max_ns = timeframe.own_ns
            + timeframe
                .children
                .iter()
                .map(|&child| ring[hierarchy[child as usize].compute_tree.get()].compute_ns)
                .max()
                .unwrap_or(0);
        *compute_parallelism.entry(timeframe.dep_kind).or_default() +=
            compute_sum_ns - compute_max_ns;
        if i % 200 == 0 {
            eprintln!(
                "{i}/{}, compute_arena: {}MiB",
                hierarchy.len(),
                mem::size_of::<ComputeNode>() * ring.len() / 1024 / 1024
            );
        }
    }

    let compute_kinds: BTreeMap<_, _> =
        compute_parallelism.into_iter().map(|(k, c)| (c, k)).collect();

    {
        // dump a .txt file with just the edges:
        let txt_path = format!("{path}.txt");
        let mut file = File::create_buffered(&txt_path).unwrap();
        for (compute_ns, kind) in compute_kinds.into_iter().rev() {
            writeln!(file, "{kind:?},{compute_ns}").unwrap();
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
