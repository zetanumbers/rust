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
        realtime: u32,
        own: u32,
        children: FxHashSet<u32>,
        parents: FxHashSet<u32>,
        depth: Cell<u32>,
        compute_tree: Cell<ComputeIdx>,
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

    #[cfg(debug_assertions)]
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
        #[inline]
        const fn null() -> ComputeIdx {
            // Has to be top element
            ComputeIdx(u32::MAX)
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
            &self.data[index.0 as usize]
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
            if self.data.try_reserve(1).is_err() {
                self.data = Vec::new();
                panic!("Out of memory!");
            }
            let new_idx = ComputeIdx(u32::try_from(self.data.len()).unwrap());
            self.data.push(node);
            // if self.has_duplicates(new_idx) {
            //     panic!(
            //         "Found duplicated nodes:\n  node: {node:x?}\n  left: {:x?},\n  right: {:x?},",
            //         self[node.left], self[node.right],
            //     )
            // }
            new_idx
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
            if lhs.is_either_null(rhs) {
                return ComputeIdx(lhs.0 & rhs.0);
            }
            if lhs == rhs {
                return lhs;
            }
            let lhs_node = self[lhs];
            let rhs_node = self[rhs];

            // struct DebugNodes {
            //     lhs_node: ComputeNode,
            //     lhs_left_right: Option<[ComputeNode; 2]>,
            //     rhs_node: ComputeNode,
            //     rhs_left_right: Option<[ComputeNode; 2]>,
            // }

            // impl Drop for DebugNodes {
            //     fn drop(&mut self) {
            //         if std::thread::panicking() {
            //             eprintln!(
            //                 "=======================\n  lhs: {:x?},\n  lhs_left: {:x?},\n  lhs_right: {:x?},\n  rhs: {:x?},\n  rhs_left: {:x?},\n  rhs_right: {:x?}",
            //                 self.lhs_node,
            //                 self.lhs_left_right.map(|a| a[0]),
            //                 self.lhs_left_right.map(|a| a[1]),
            //                 self.rhs_node,
            //                 self.rhs_left_right.map(|a| a[0]),
            //                 self.rhs_left_right.map(|a| a[1]),
            //             )
            //         }
            //     }
            // }

            // let _g = DebugNodes {
            //     lhs_node,
            //     rhs_node,
            //     lhs_left_right: self
            //         .data
            //         .get(lhs_node.left.0 as usize)
            //         .map(|l| [*l, self[lhs_node.right]]),
            //     rhs_left_right: self
            //         .data
            //         .get(rhs_node.left.0 as usize)
            //         .map(|l| [*l, self[rhs_node.right]]),
            // };

            match lhs_node.neighborhood.cmp_or_union(rhs_node.neighborhood) {
                Ok(cmp::Ordering::Equal) => {
                    let left = self.union(lhs_node.left, rhs_node.left);
                    let right = self.union(lhs_node.right, rhs_node.right);
                    debug_assert!(!left.is_either_null(right) || left.is_both_null(right));
                    if [left, right] == [lhs_node.left, lhs_node.right] {
                        return lhs;
                    }
                    if [left, right] == [rhs_node.left, rhs_node.right] {
                        return rhs;
                    }
                    self.alloc_node(ComputeNode {
                        left,
                        right,
                        neighborhood: lhs_node.neighborhood,
                        compute: self[left].compute + self[right].compute,
                    })
                }
                Err(disjoint_union) => {
                    let [left, right] = if lhs_node.neighborhood.0 < rhs_node.neighborhood.0 {
                        [lhs, rhs]
                    } else {
                        [rhs, lhs]
                    };
                    self.alloc_node(ComputeNode {
                        left,
                        right,
                        neighborhood: disjoint_union,
                        compute: lhs_node.compute + rhs_node.compute,
                    })
                }
                Ok(cmp::Ordering::Less) => {
                    self.union_included(rhs, &rhs_node, lhs, lhs_node.neighborhood)
                }
                Ok(cmp::Ordering::Greater) => {
                    self.union_included(lhs, &lhs_node, rhs, rhs_node.neighborhood)
                }
            }
        }

        fn union_included(
            &mut self,
            larger: ComputeIdx,
            larger_node: &ComputeNode,
            smaller: ComputeIdx,
            smaller_neighborhood: IndexNeighborhood,
        ) -> ComputeIdx {
            debug_assert!(smaller_neighborhood < larger_node.neighborhood);
            let [left, right];
            if smaller_neighborhood.0 <= larger_node.neighborhood.0 {
                left = self.union(smaller, larger_node.left);
                if larger_node.left == left {
                    return larger;
                }
                right = larger_node.right;
            } else {
                right = self.union(smaller, larger_node.right);
                if larger_node.right == right {
                    return larger;
                }
                left = larger_node.left;
            };
            let node = ComputeNode {
                left,
                right,
                neighborhood: larger_node.neighborhood,
                compute: self[left].compute + self[right].compute,
            };
            self.alloc_node(node)
        }

        // fn has_duplicates(&mut self, tree: ComputeIdx) -> bool {
        //     struct CheckIndices<'a> {
        //         indices: FxHashSet<IndexNeighborhood>,
        //         ring: &'a ComputeRing,
        //     }

        //     impl CheckIndices<'_> {
        //         fn check(&mut self, node: ComputeIdx) -> bool {
        //             let node = &self.ring[node];
        //             !self.indices.insert(node.neighborhood)
        //                 || if node.neighborhood.0 % 2 != 0 {
        //                     debug_assert_ne!(node.left, ComputeIdx::null());
        //                     debug_assert_ne!(node.right, ComputeIdx::null());
        //                     self.check(node.left) || self.check(node.right)
        //                 } else {
        //                     false
        //                 }
        //         }
        //     }

        //     let mut check_indices = CheckIndices { indices: FxHashSet::default(), ring: self };
        //     check_indices.check(tree)
        // }
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
                // quantize by 16ns
                u32::try_from(node.data.timeframe.as_nanos() as u64 / 16).unwrap()
            } else {
                0
            };
            (
                (node.data.inner, idx.node_id() as u32),
                Timeframe {
                    dep_kind: node.data.inner.kind,
                    realtime,
                    own: realtime,
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
            let child_realtime_ns = child.realtime;
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
            if let Some(new_own) = parent.own.checked_sub(substract_ns) {
                parent.own = new_own;
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
                    realtime: old.realtime,
                    own: old.own,
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

        let this_leaf = ring.alloc_leaf_node(timeframe.own, i);
        let this_tree = timeframe.children.iter().fold(this_leaf, |acc, &child| {
            let child_tree = hierarchy[child as usize].compute_tree.get();
            debug_assert_ne!(child_tree, ComputeIdx::null());
            ring.union(acc, child_tree)
        });
        timeframe.compute_tree.set(this_tree);

        let compute_sum = ring[this_tree].compute;
        let timeframe = &hierarchy[i as usize];
        let compute_max = timeframe.own
            + timeframe
                .children
                .iter()
                .map(|&child| ring[hierarchy[child as usize].compute_tree.get()].compute)
                .max()
                .unwrap_or(0);
        *compute_parallelism.entry(timeframe.dep_kind).or_default() +=
            (compute_sum - compute_max) as u64;
        if i % 1024 == 0 {
            eprintln!("{i}/{}", hierarchy.len());
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
