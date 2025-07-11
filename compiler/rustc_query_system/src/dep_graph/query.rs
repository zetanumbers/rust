use std::time::Duration;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph::linked_graph::{Direction, INCOMING, LinkedGraph, NodeIndex};
use rustc_index::IndexVec;

use super::{DepNode, DepNodeIndex};
use crate::dep_graph::DepCache;

#[derive(Debug, Clone, Copy)]
pub struct TimedDepNode {
    pub inner: DepNode,
    pub timeframe: Duration,
}

pub struct DepGraphQuery {
    pub graph: LinkedGraph<TimedDepNode, DepCache>,
    pub indices: FxHashMap<DepNode, NodeIndex>,
    pub dep_index_to_index: IndexVec<DepNodeIndex, Option<NodeIndex>>,
}

impl DepGraphQuery {
    pub fn new(prev_node_count: usize) -> DepGraphQuery {
        let node_count = prev_node_count + prev_node_count / 4;
        let edge_count = 6 * node_count;

        let graph = LinkedGraph::with_capacity(node_count, edge_count);
        let indices = FxHashMap::default();
        let dep_index_to_index = IndexVec::new();

        DepGraphQuery { graph, indices, dep_index_to_index }
    }

    pub fn push(
        &mut self,
        index: DepNodeIndex,
        node: DepNode,
        timeframe: Duration,
        edges: &[(DepNodeIndex, DepCache)],
    ) {
        let source = self.graph.add_node(TimedDepNode { inner: node, timeframe });
        self.dep_index_to_index.insert(index, source);
        self.indices.insert(node, source);

        for &(target, dep_cache) in edges.iter() {
            let target = self.dep_index_to_index[target];
            // We may miss the edges that are pushed while the `DepGraphQuery` is being accessed.
            // Skip them to issues.
            if let Some(target) = target {
                self.graph.add_edge(source, target, dep_cache);
            }
        }
    }

    pub fn nodes(&self) -> Vec<&DepNode> {
        self.graph.all_nodes().iter().map(|n| &n.data.inner).collect()
    }

    pub fn edges(&self) -> Vec<(&DepNode, &DepNode)> {
        self.graph
            .all_edges()
            .iter()
            .map(|edge| (edge.source(), edge.target()))
            .map(|(s, t)| (&self.graph.node_data(s).inner, &self.graph.node_data(t).inner))
            .collect()
    }

    fn reachable_nodes(&self, node: &DepNode, direction: Direction) -> Vec<&DepNode> {
        if let Some(&index) = self.indices.get(node) {
            self.graph
                .depth_traverse(index, direction)
                .map(|s| &self.graph.node_data(s).inner)
                .collect()
        } else {
            vec![]
        }
    }

    /// All nodes that can reach `node`.
    pub fn transitive_predecessors(&self, node: &DepNode) -> Vec<&DepNode> {
        self.reachable_nodes(node, INCOMING)
    }
}
