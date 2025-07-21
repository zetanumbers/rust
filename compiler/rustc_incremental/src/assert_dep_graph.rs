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
use std::{cmp, env, mem};

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_data_structures::graph::linked_graph::{Direction, INCOMING, NodeIndex, OUTGOING};
use rustc_data_structures::stack::ensure_sufficient_stack;
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
        compute_tree_idx: Cell<usize>,
    }

    #[derive(Debug, Clone, Copy)]
    struct ComputeNode {
        left: usize,
        right: usize,
        shared_high_bits: u32,
        shared_high_bitmask: u32,
        compute_ns: u64,
    }

    impl ComputeNode {
        // Has to be top element
        const NULL_IDX: usize = usize::MAX;

        fn alloc_leaf_node(
            own_ns: u64,
            timeframe_idx: u32,
            compute_arena: &mut Vec<ComputeNode>,
        ) -> usize {
            let new_idx = compute_arena.len();
            compute_arena.push(ComputeNode {
                left: ComputeNode::NULL_IDX,
                right: ComputeNode::NULL_IDX,
                compute_ns: own_ns,
                shared_high_bitmask: !0,
                shared_high_bits: timeframe_idx,
            });
            new_idx
        }

        fn union(mut lhs: usize, mut rhs: usize, compute_arena: &mut Vec<ComputeNode>) -> usize {
            if lhs | rhs == ComputeNode::NULL_IDX {
                return lhs & rhs;
            }
            if lhs == rhs {
                return lhs;
            }
            let mut lhs_node = compute_arena[lhs];
            let mut rhs_node = compute_arena[rhs];
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
                        let left = ComputeNode::union(lhs_node.left, rhs_node.left, compute_arena);
                        let right =
                            ComputeNode::union(lhs_node.right, rhs_node.right, compute_arena);
                        if [left, right] == [lhs_node.left, lhs_node.right] {
                            return lhs;
                        }

                        debug_assert!(
                            left | right != ComputeNode::NULL_IDX
                                || left & right == ComputeNode::NULL_IDX
                        );

                        let idx = compute_arena.len();
                        compute_arena.push(ComputeNode {
                            left,
                            right,
                            shared_high_bits: lhs_node.shared_high_bits,
                            shared_high_bitmask: lhs_node.shared_high_bitmask,
                            compute_ns: compute_arena[left].compute_ns
                                + compute_arena[right].compute_ns,
                        });
                        return idx;
                    } else {
                        let idx = compute_arena.len();
                        compute_arena.push(ComputeNode {
                            left,
                            right,
                            shared_high_bits: lhs_node.shared_high_bits & shared_high_bitmask,
                            shared_high_bitmask,
                            compute_ns: lhs_node.compute_ns + rhs_node.compute_ns,
                        });
                        return idx;
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
                let idx = compute_arena.len();
                compute_arena.push(ComputeNode {
                    left,
                    right,
                    shared_high_bits: lhs_node.shared_high_bits & shared_high_bitmask,
                    shared_high_bitmask,
                    compute_ns: lhs_node.compute_ns + rhs_node.compute_ns,
                });
                idx
            } else {
                let [left, right] = if lhs_node.shared_high_bits
                    <= rhs_node.shared_high_bits | (!rhs_node.shared_high_bitmask).wrapping_shr(1)
                {
                    let left = ComputeNode::union(lhs_node.left, rhs, compute_arena);
                    if left == lhs_node.left {
                        return lhs;
                    }
                    [left, lhs_node.right]
                } else {
                    let right = ComputeNode::union(lhs_node.right, rhs, compute_arena);
                    if right == lhs_node.right {
                        return rhs;
                    }
                    [lhs_node.left, right]
                };

                debug_assert!(
                    left | right != ComputeNode::NULL_IDX || left & right == ComputeNode::NULL_IDX
                );

                let idx = compute_arena.len();
                let get_compute_ns = |idx: usize| {
                    if idx != ComputeNode::NULL_IDX { compute_arena[idx].compute_ns } else { 0 }
                };
                compute_arena.push(ComputeNode {
                    left,
                    right,
                    shared_high_bits: lhs_node.shared_high_bits,
                    shared_high_bitmask: lhs_node.shared_high_bitmask,
                    compute_ns: get_compute_ns(left) + get_compute_ns(right),
                });
                idx
            }
        }
    }

    assert!(query.graph.len_nodes() <= u32::MAX as usize);
    let mut side_effect_node_count = 0;
    let (nodes_to_indices, mut hierarchy): (FxHashMap<_, u32>, Vec<_>) = query
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
                    compute_tree_idx: Cell::new(ComputeNode::NULL_IDX),
                },
            )
        })
        .collect();
    debug_assert_eq!(nodes_to_indices.len() + side_effect_node_count - 1, hierarchy.len());

    with_context(|icx| {
        let tcx = icx.tcx;
        for edge in query.graph.all_edges() {
            let child_idx = edge.target().node_id();
            let child = &hierarchy[child_idx];
            let child_realtime_ns = child.realtime_ns;
            let substract_ns = match edge.data {
                DepCache::Computed => child_realtime_ns,
                DepCache::Cached => 0,
            };
            let parent = &mut hierarchy[edge.source().node_id()];
            if !parent.children.insert(child_idx as u32) {
                continue;
            }
            if let Some(new_own) = parent.own_ns.checked_sub(substract_ns) {
                parent.own_ns = new_own;
            } else {
                assert!(tcx.dep_kind_info(parent.dep_kind).is_anon)
            }
        }
    });

    let mut compute_arena = Vec::with_capacity(query.graph.len_edges());
    let mut compute_kinds = FxHashMap::<DepKind, u64>::default();

    for i in 0..hierarchy.len() as u32 {
        fn calc_compute_tree(
            i: u32,
            hierarchy: &[Timeframe],
            compute_arena: &mut Vec<ComputeNode>,
        ) -> usize {
            let timeframe = &hierarchy[i as usize];
            let this_tree = timeframe.compute_tree_idx.get();
            if this_tree != ComputeNode::NULL_IDX {
                return this_tree;
            }

            let this_leaf = ComputeNode::alloc_leaf_node(timeframe.own_ns, i, compute_arena);
            let this_tree = ensure_sufficient_stack(|| {
                timeframe.children.iter().fold(this_leaf, |acc, &child| {
                    ComputeNode::union(
                        acc,
                        calc_compute_tree(child, hierarchy, compute_arena),
                        compute_arena,
                    )
                })
            });
            timeframe.compute_tree_idx.set(this_tree);
            this_tree
        }

        let compute_tree = calc_compute_tree(i, &hierarchy, &mut compute_arena);
        let compute_sum_ns = compute_arena[compute_tree].compute_ns;
        let timeframe = &hierarchy[i as usize];
        let compute_max_ns = timeframe.own_ns
            + timeframe
                .children
                .iter()
                .map(|&child| {
                    compute_arena[hierarchy[child as usize].compute_tree_idx.get()].compute_ns
                })
                .max()
                .unwrap_or(0);
        *compute_kinds.entry(timeframe.dep_kind).or_default() += compute_sum_ns - compute_max_ns;
        if i % 100 == 0 {
            eprintln!(
                "{i}/{}, compute_arena: {}MiB",
                hierarchy.len(),
                mem::size_of::<ComputeNode>() * compute_arena.len() / 1024 / 1024
            );
        }
    }

    let compute_kinds: BTreeMap<_, _> = compute_kinds.into_iter().map(|(k, c)| (c, k)).collect();

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
