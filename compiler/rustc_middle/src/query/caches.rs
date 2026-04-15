use std::sync::OnceLock;

use rustc_data_structures::hash_table::{self, HashTable};
use rustc_data_structures::sharded::{Sharded, make_hash};
pub use rustc_data_structures::vec_cache::VecCache;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_index::Idx;
use rustc_span::def_id::{DefId, DefIndex};

use crate::dep_graph::DepNodeIndex;
use crate::query::keys::QueryKey;

/// Trait for types that serve as an in-memory cache for query results,
/// for a given key (argument) type and value (return) type.
///
/// Types implementing this trait are associated with actual key/value types
/// by the `Cache` associated type of the `rustc_middle::query::Key` trait.
pub trait QueryCache: Sized {
    type Key: QueryKey;
    type Value: Copy;

    /// Returns the cached value (and other information) associated with the
    /// given key, if it is present in the cache.
    fn lookup(&self, key: &Self::Key) -> Option<(Self::Value, DepNodeIndex)>;

    /// Adds a key/value entry to this cache.
    ///
    /// Called by some part of the query system, after having obtained the
    /// value by executing the query or loading a cached value from disk.
    fn complete(&self, key: Self::Key, value: Self::Value, index: DepNodeIndex);

    /// Calls a closure on each entry in this cache.
    fn for_each(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex));

    /// Returns the number of entries currently in this cache.
    ///
    /// Useful for reserving capacity in data structures that will hold the
    /// output of a call to [`Self::for_each`].
    fn len(&self) -> usize;
}

/// In-memory cache for queries whose keys aren't suitable for any of the
/// more specialized kinds of cache. Backed by a sharded hashmap.
pub struct DefaultCache<K, V> {
    inner: Sharded<DefaultCacheShard<K, V>>,
}

struct DefaultCacheShard<K, V> {
    indices: HashTable<(K, u32)>,
    // arena for temporal cache locality
    arena: Vec<(V, DepNodeIndex)>,
}

impl<K, V> Default for DefaultCacheShard<K, V> {
    fn default() -> Self {
        DefaultCacheShard { indices: HashTable::with_capacity(32), arena: Vec::with_capacity(32) }
    }
}

impl<K, V> Default for DefaultCache<K, V> {
    fn default() -> Self {
        DefaultCache { inner: Default::default() }
    }
}

impl<K, V> QueryCache for DefaultCache<K, V>
where
    K: QueryKey,
    V: Copy,
{
    type Key = K;
    type Value = V;

    fn lookup(&self, key: &K) -> Option<(V, DepNodeIndex)> {
        let hash = make_hash(&key);
        let shard = self.inner.lock_shard_by_hash(hash);
        // SAFETY: we allocate on arena before adding index to the hashmap cache
        Some(*unsafe {
            shard.arena.get_unchecked(shard.indices.find(hash, |x| x.0 == *key)?.1 as usize)
        })
    }

    fn complete(&self, key: K, value: V, index: DepNodeIndex) {
        // We may be overwriting another value. This is all right, since the dep-graph
        // will check that the value fingerprint matches.

        let hash = make_hash(&key);
        let mut shard = self.inner.lock_shard_by_hash(hash);

        let i = shard.arena.len() as u32;
        shard.arena.push((value, index));

        if cfg!(debug_assertions) {
            match shard.indices.entry(hash, |(k, _)| *k == key, |(k, _)| make_hash(k)) {
                hash_table::Entry::Occupied(_) => {
                    panic!("query cache entry is already occupied");
                }
                hash_table::Entry::Vacant(e) => {
                    e.insert((key, i));
                }
            }
        } else {
            shard.indices.insert_unique(hash, (key, i), |(k, _)| make_hash(k));
        }
    }

    fn for_each(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        for shard in self.inner.lock_shards() {
            for (k, i) in shard.indices.iter() {
                // SAFETY: values are stored in arena and then indices are saved while shard is locked
                let v = unsafe { &shard.arena.get(*i as usize).unwrap_unchecked() };
                f(k, &v.0, v.1);
            }
        }
    }

    fn len(&self) -> usize {
        self.inner.lock_shards().map(|shard| shard.indices.len()).sum()
    }
}

/// In-memory cache for queries whose key type only has one value (e.g. `()`).
/// The cache therefore only needs to store one query return value.
pub struct SingleCache<V> {
    cache: OnceLock<(V, DepNodeIndex)>,
}

impl<V> Default for SingleCache<V> {
    fn default() -> Self {
        SingleCache { cache: OnceLock::new() }
    }
}

impl<V> QueryCache for SingleCache<V>
where
    V: Copy,
{
    type Key = ();
    type Value = V;

    #[inline(always)]
    fn lookup(&self, _key: &()) -> Option<(V, DepNodeIndex)> {
        self.cache.get().copied()
    }

    #[inline]
    fn complete(&self, _key: (), value: V, index: DepNodeIndex) {
        self.cache.set((value, index)).ok();
    }

    fn for_each(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        if let Some(value) = self.cache.get() {
            f(&(), &value.0, value.1)
        }
    }

    fn len(&self) -> usize {
        self.cache.get().is_some().into()
    }
}

/// In-memory cache for queries whose key is a [`DefId`].
///
/// Selects between one of two internal caches, depending on whether the key
/// is a local ID or foreign-crate ID.
pub struct DefIdCache<V> {
    /// Stores the local DefIds in a dense map. Local queries are much more often dense, so this is
    /// a win over hashing query keys at marginal memory cost (~5% at most) compared to FxHashMap.
    local: VecCache<DefIndex, V, DepNodeIndex>,
    foreign: DefaultCache<DefId, V>,
}

impl<V> Default for DefIdCache<V> {
    fn default() -> Self {
        DefIdCache { local: Default::default(), foreign: Default::default() }
    }
}

impl<V> QueryCache for DefIdCache<V>
where
    V: Copy,
{
    type Key = DefId;
    type Value = V;

    #[inline(always)]
    fn lookup(&self, key: &DefId) -> Option<(V, DepNodeIndex)> {
        if key.krate == LOCAL_CRATE {
            self.local.lookup(&key.index)
        } else {
            self.foreign.lookup(key)
        }
    }

    #[inline]
    fn complete(&self, key: DefId, value: V, index: DepNodeIndex) {
        if key.krate == LOCAL_CRATE {
            self.local.complete(key.index, value, index)
        } else {
            self.foreign.complete(key, value, index)
        }
    }

    fn for_each(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        self.local.for_each(&mut |key, value, index| {
            f(&DefId { krate: LOCAL_CRATE, index: *key }, value, index);
        });
        self.foreign.for_each(f);
    }

    fn len(&self) -> usize {
        self.local.len() + self.foreign.len()
    }
}

impl<K, V> QueryCache for VecCache<K, V, DepNodeIndex>
where
    K: Idx + QueryKey,
    V: Copy,
{
    type Key = K;
    type Value = V;

    #[inline(always)]
    fn lookup(&self, key: &K) -> Option<(V, DepNodeIndex)> {
        self.lookup(key)
    }

    #[inline]
    fn complete(&self, key: K, value: V, index: DepNodeIndex) {
        self.complete(key, value, index)
    }

    fn for_each(&self, f: &mut dyn FnMut(&Self::Key, &Self::Value, DepNodeIndex)) {
        self.for_each(f)
    }

    fn len(&self) -> usize {
        self.len()
    }
}
