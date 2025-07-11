use std::hash::{Hash, Hasher};
use std::ops::Deref;

use rustc_index::bit_set::GrowableBitSet;
use smallvec::SmallVec;

use crate::dep_graph::DepNodeIndex;

#[derive(Default, Debug)]
pub(crate) struct EdgesVec {
    max: u32,
    edges: SmallVec<[DepNodeIndex; EdgesVec::INLINE_CAPACITY]>,
    cached: GrowableBitSet<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeCache {
    Cached,
    Computed,
}

impl Hash for EdgesVec {
    #[inline]
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        Hash::hash(&self.edges, hasher)
    }
}

impl EdgesVec {
    pub(crate) const INLINE_CAPACITY: usize = 8;

    #[inline]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub(crate) fn push(&mut self, edge: DepNodeIndex, cache: EdgeCache) {
        self.max = self.max.max(edge.as_u32());
        self.edges.push(edge);
        if let EdgeCache::Cached = cache {
            let last = self.edges.len();
            self.cached.ensure(last);
            self.cached.insert(last);
        }
    }

    #[inline]
    pub(crate) fn max_index(&self) -> u32 {
        self.max
    }

    pub(crate) fn extend_from_other(&mut self, other: &EdgesVec) {
        self.max = self.max.max(other.max);
        let append_base = self.edges.len();

        self.edges.extend_from_slice(&other.edges);

        self.cached.ensure(self.edges.len());
        for i in other.cached.iter() {
            self.cached.insert(append_base + i);
        }
    }

    pub(crate) fn cache(&self, i: usize) -> EdgeCache {
        if self.cached.contains(i) { EdgeCache::Cached } else { EdgeCache::Computed }
    }
}

impl Deref for EdgesVec {
    type Target = [DepNodeIndex];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.edges.as_slice()
    }
}

impl FromIterator<(DepNodeIndex, EdgeCache)> for EdgesVec {
    #[inline]
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (DepNodeIndex, EdgeCache)>,
    {
        let mut vec = EdgesVec::new();
        for (index, cache) in iter {
            vec.push(index, cache);
        }
        vec
    }
}

impl Extend<(DepNodeIndex, EdgeCache)> for EdgesVec {
    #[inline]
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = (DepNodeIndex, EdgeCache)>,
    {
        for (index, cache) in iter {
            self.push(index, cache);
        }
    }
}
