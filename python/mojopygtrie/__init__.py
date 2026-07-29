"""A Mojo-backed trie with the public API of pygtrie 2.5."""

from __future__ import annotations

import collections.abc
import copy as _copy

import numpy as np

from ._lib import addr, lib


class ShortKeyError(KeyError):
    pass


class _Empty:
    pass


_EMPTY = _Empty()


def _i64_filled(size: int, value: int) -> np.ndarray:
    array = np.empty(size, dtype=np.int64)
    if size:
        lib().mpg_fill_i64(addr(array), size, value)
    return array


class _ChildrenIterator:
    def __init__(self, factory):
        self._factory = factory
        self._iterator = None

    def __bool__(self):
        return True

    def __iter__(self):
        if self._iterator is None:
            self._iterator = iter(self._factory())
        return self._iterator

    def __next__(self):
        return next(iter(self))


class KeyBatch:
    """Pre-encoded keys reusable by bulk lookup methods."""

    __slots__ = ("_owner", "_generation", "keys", "tokens", "offsets")

    def __init__(self, owner, generation, keys, tokens, offsets):
        self._owner = owner
        self._generation = generation
        self.keys = keys
        self.tokens = tokens
        self.offsets = offsets

    def __len__(self):
        return len(self.keys)


class Trie(collections.abc.MutableMapping):
    HAS_VALUE = 1
    HAS_SUBTRIE = 2
    _raw_keys_are_cache_keys = False

    def __init__(self, *args, **kwargs):
        self._sorted = False
        self._token_by_step = {}
        self._step_by_token = []
        self._generation = 0
        self._reset_storage(reset_tokens=False)
        self.update(*args, **kwargs)

    def _reset_storage(self, reset_tokens=True):
        if reset_tokens:
            self._token_by_step.clear()
            self._step_by_token.clear()
            self._generation += 1
        self._node_by_path = {}
        capacity = 16
        self._first = _i64_filled(capacity, -1)
        self._last = _i64_filled(capacity, -1)
        self._parent = _i64_filled(capacity, -1)
        self._parent_token = _i64_filled(capacity, -1)
        self._edge_token = _i64_filled(capacity, -1)
        self._edge_child = _i64_filled(capacity, -1)
        self._edge_next = _i64_filled(capacity, -1)
        self._has_value = np.zeros(capacity, dtype=np.uint8)
        self._values = [_EMPTY] * capacity
        self._hash_nodes = _i64_filled(32, -1)
        self._hash_tokens = _i64_filled(32, -1)
        self._hash_children = _i64_filled(32, -1)
        self._state = np.array([1, 0], dtype=np.int64)
        self._hash_used = 0
        self._size = 0

    @staticmethod
    def _grow(array, size, fill):
        grown = np.empty(size, dtype=array.dtype)
        if array.dtype == np.int64:
            lib().mpg_grow_i64(addr(array), addr(grown), array.size, size, fill)
        else:
            grown[: array.size] = array
            grown[array.size :] = fill
        return grown

    def _ensure_capacity(self, maximum_new_edges):
        needed = int(self._state[1]) + maximum_new_edges
        if needed > self._edge_token.size:
            size = max(needed, self._edge_token.size * 2)
            self._edge_token = self._grow(self._edge_token, size, -1)
            self._edge_child = self._grow(self._edge_child, size, -1)
            self._edge_next = self._grow(self._edge_next, size, -1)
        node_needed = int(self._state[0]) + maximum_new_edges
        if node_needed > self._first.size:
            old = self._first.size
            size = max(node_needed, old * 2)
            self._first = self._grow(self._first, size, -1)
            self._last = self._grow(self._last, size, -1)
            self._parent = self._grow(self._parent, size, -1)
            self._parent_token = self._grow(self._parent_token, size, -1)
            self._has_value = self._grow(self._has_value, size, 0)
            self._values.extend([_EMPTY] * (size - old))
        if (self._hash_used + maximum_new_edges) * 10 >= self._hash_nodes.size * 7:
            size = self._hash_nodes.size
            while (self._hash_used + maximum_new_edges) * 10 >= size * 7:
                size *= 2
            self._rehash(size)

    def _rehash(self, capacity):
        nodes = _i64_filled(capacity, -1)
        tokens = _i64_filled(capacity, -1)
        children = _i64_filled(capacity, -1)
        used = 0
        stack = [0]
        while stack:
            node = stack.pop()
            edge = int(self._first[node])
            while edge >= 0:
                token = int(self._edge_token[edge])
                child = int(self._edge_child[edge])
                slot = (node * 1000003 + token * 9176) & (capacity - 1)
                while nodes[slot] >= 0:
                    slot = (slot + 1) & (capacity - 1)
                nodes[slot], tokens[slot], children[slot] = node, token, child
                used += 1
                stack.append(child)
                edge = int(self._edge_next[edge])
        self._hash_nodes = nodes
        self._hash_tokens = tokens
        self._hash_children = children
        self._hash_used = used

    def _path_from_key(self, key):
        return key

    def _key_from_path(self, path):
        return tuple(path)

    def _path(self, key):
        return () if key is _EMPTY else tuple(self._path_from_key(key))

    def _encode_path(self, path, create):
        encoded = np.empty(len(path), dtype=np.int64)
        for index, step in enumerate(path):
            token = self._token_by_step.get(step, -1)
            if token < 0:
                if not create:
                    return None
                token = len(self._step_by_token)
                self._token_by_step[step] = token
                self._step_by_token.append(step)
            encoded[index] = token
        return encoded

    def _encode_paths(self, paths, create):
        offsets = np.empty(len(paths) + 1, dtype=np.int64)
        flat = []
        append = flat.append
        token_by_step = self._token_by_step
        step_by_token = self._step_by_token
        for index, path in enumerate(paths):
            offsets[index] = len(flat)
            for step in path:
                token = token_by_step.get(step, -1)
                if token < 0:
                    if not create:
                        append(-1)
                        break
                    token = len(step_by_token)
                    token_by_step[step] = token
                    step_by_token.append(step)
                append(token)
        offsets[-1] = len(flat)
        return np.asarray(flat, dtype=np.int64), offsets

    def _encode(self, key, create=False):
        path = self._path(key)
        return path, self._encode_path(path, create)

    def _cache_key(self, key, path):
        return path

    def _cache_key_without_path(self, key):
        return None

    def _find_encoded(self, encoded):
        if encoded is None:
            return -1
        if not encoded.size:
            return 0
        return int(
            lib().mpg_find(
                addr(self._hash_nodes),
                addr(self._hash_tokens),
                addr(self._hash_children),
                self._hash_nodes.size,
                addr(encoded),
                encoded.size,
            )
        )

    def _find_node(self, key):
        cache_key = self._cache_key_without_path(key)
        if cache_key is not None:
            try:
                return self._node_by_path[cache_key]
            except KeyError:
                pass
        path = self._path(key)
        if cache_key is None:
            cache_key = self._cache_key(key, path)
            try:
                return self._node_by_path[cache_key]
            except KeyError:
                pass
        encoded = self._encode_path(path, create=False)
        node = self._find_encoded(encoded)
        if node < 0:
            raise KeyError(key)
        return node

    def _insert_encoded(self, encoded):
        if not encoded.size:
            return 0
        self._ensure_capacity(encoded.size)
        old_edges = int(self._state[1])
        node = int(
            lib().mpg_insert(
                addr(self._hash_nodes),
                addr(self._hash_tokens),
                addr(self._hash_children),
                self._hash_nodes.size,
                addr(self._first),
                addr(self._last),
                addr(self._edge_token),
                addr(self._edge_child),
                addr(self._edge_next),
                addr(self._parent),
                addr(self._parent_token),
                addr(self._state),
                addr(encoded),
                encoded.size,
            )
        )
        self._hash_used += int(self._state[1]) - old_edges
        return node

    def _assign_node(self, node, value, only_if_missing=False):
        if not self._has_value[node]:
            self._has_value[node] = 1
            self._size += 1
            self._values[node] = value
        elif not only_if_missing:
            self._values[node] = value
        return self._values[node]

    def _set_node(self, key, value, only_if_missing=False):
        path = self._path(key)
        encoded = self._encode_path(path, create=True)
        node = self._insert_encoded(encoded)
        self._node_by_path[self._cache_key(key, path)] = node
        self._assign_node(node, value, only_if_missing)
        return node

    def _set_many(self, pairs):
        if not pairs:
            return
        paths = [self._path(key) for key, _ in pairs]
        flat, offsets = self._encode_paths(paths, create=True)
        self._ensure_capacity(flat.size)
        result = np.empty(len(pairs), dtype=np.int64)
        # All-empty paths have no token storage, but the Mojo ABI uses
        # non-nullable pointers even when every [start, stop) range is empty.
        ffi_flat = flat if flat.size else np.zeros(1, dtype=np.int64)
        old_edges = int(self._state[1])
        lib().mpg_bulk_insert(
            addr(self._hash_nodes),
            addr(self._hash_tokens),
            addr(self._hash_children),
            self._hash_nodes.size,
            addr(self._first),
            addr(self._last),
            addr(self._edge_token),
            addr(self._edge_child),
            addr(self._edge_next),
            addr(self._parent),
            addr(self._parent_token),
            addr(self._state),
            addr(ffi_flat),
            addr(offsets),
            len(pairs),
            addr(result),
        )
        self._hash_used += int(self._state[1]) - old_edges
        for node, path, (key, value) in zip(result, paths, pairs):
            node = int(node)
            self._node_by_path[self._cache_key(key, path)] = node
            self._assign_node(node, value)

    def update(self, *args, **kwargs):
        if len(args) > 1:
            raise ValueError(
                f"update() takes at most one positional argument, {len(args)} given."
            )
        pairs = []
        if args:
            source = args[0]
            if hasattr(source, "keys"):
                pairs.extend((key, source[key]) for key in source.keys())
            else:
                pairs.extend(source)
        pairs.extend(kwargs.items())
        self._set_many(pairs)

    @classmethod
    def fromkeys(cls, keys, value=None):
        trie = cls()
        trie._set_many([(key, value) for key in keys])
        return trie

    def clear(self):
        self._reset_storage()

    def enable_sorting(self, enable=True):
        self._sorted = bool(enable)

    def __getitem__(self, key_or_slice):
        key, is_slice = self._slice_maybe(key_or_slice)
        if is_slice:
            return self.itervalues(prefix=key)
        node = self._find_node(key)
        if not self._has_value[node]:
            raise ShortKeyError(key)
        return self._values[node]

    def __setitem__(self, key_or_slice, value):
        key, is_slice = self._slice_maybe(key_or_slice)
        node = self._set_node(key, value)
        if is_slice:
            self._drop_children(node)

    def setdefault(self, key, default=None):
        node = self._set_node(key, default, only_if_missing=True)
        return self._values[node]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    @staticmethod
    def _slice_maybe(key_or_slice):
        if isinstance(key_or_slice, slice):
            if key_or_slice.stop is not None or key_or_slice.step is not None:
                raise TypeError(key_or_slice)
            return key_or_slice.start, True
        return key_or_slice, False

    def _hash_delete(self, parent, token):
        capacity = self._hash_nodes.size
        slot = (parent * 1000003 + token * 9176) & (capacity - 1)
        for _ in range(capacity):
            owner = int(self._hash_nodes[slot])
            if owner == -1:
                return
            if owner == parent and int(self._hash_tokens[slot]) == token:
                self._hash_nodes[slot] = -2
                self._hash_used -= 1
                return
            slot = (slot + 1) & (capacity - 1)

    def _unlink_child(self, parent, token):
        self._node_by_path.clear()
        self._hash_delete(parent, token)
        previous = -1
        edge = int(self._first[parent])
        while edge >= 0 and int(self._edge_token[edge]) != token:
            previous, edge = edge, int(self._edge_next[edge])
        if edge < 0:
            return
        following = int(self._edge_next[edge])
        if previous < 0:
            self._first[parent] = following
        else:
            self._edge_next[previous] = following
        if int(self._last[parent]) == edge:
            self._last[parent] = previous

    def _prune(self, node):
        while node and not self._has_value[node] and self._first[node] < 0:
            parent = int(self._parent[node])
            self._unlink_child(parent, int(self._parent_token[node]))
            node = parent

    def _collect_nodes(self, root=0, shallow=False):
        count = int(self._state[0])
        stack = np.empty(max(count, 1), dtype=np.int64)
        result = np.empty(max(self._size, 1), dtype=np.int64)
        found = int(
            lib().mpg_collect(
                addr(self._first),
                addr(self._edge_child),
                addr(self._edge_next),
                addr(self._has_value),
                root,
                int(shallow),
                addr(stack),
                addr(result),
            )
        )
        return result[:found]

    def _drop_children(self, node):
        self._node_by_path.clear()
        edges = []
        edge = int(self._first[node])
        while edge >= 0:
            edges.append((int(self._edge_token[edge]), int(self._edge_child[edge])))
            edge = int(self._edge_next[edge])
        for token, child in edges:
            self._size -= len(self._collect_nodes(child))
            self._hash_delete(node, token)
        self._first[node] = self._last[node] = -1

    def _delete_subtree(self, node):
        removed = len(self._collect_nodes(node))
        self._size -= removed
        if node == 0:
            self._reset_storage()
            return
        parent = int(self._parent[node])
        self._unlink_child(parent, int(self._parent_token[node]))
        self._prune(parent)

    def pop(self, key, default=_EMPTY):
        try:
            node = self._find_node(key)
        except KeyError:
            if default is not _EMPTY:
                return default
            raise
        if not self._has_value[node]:
            if default is not _EMPTY:
                return default
            raise ShortKeyError(key)
        value = self._values[node]
        self._values[node] = _EMPTY
        self._has_value[node] = 0
        self._size -= 1
        self._node_by_path.clear()
        self._prune(node)
        return value

    def popitem(self):
        nodes = self._collect_nodes()
        if not len(nodes):
            raise KeyError()
        node = int(nodes[0])
        key = self._key_for_node(node)
        return key, self.pop(key)

    def __delitem__(self, key_or_slice):
        key, is_slice = self._slice_maybe(key_or_slice)
        node = self._find_node(key)
        if is_slice:
            self._delete_subtree(node)
        elif not self._has_value[node]:
            raise ShortKeyError(key)
        else:
            self.pop(key)

    def _children(self, node, sorted_children=None):
        children = []
        edge = int(self._first[node])
        while edge >= 0:
            token = int(self._edge_token[edge])
            children.append((token, int(self._edge_child[edge])))
            edge = int(self._edge_next[edge])
        if self._sorted if sorted_children is None else sorted_children:
            children.sort(key=lambda pair: self._step_by_token[pair[0]])
        return children

    def _node_path(self, node):
        path = []
        while node:
            path.append(self._step_by_token[int(self._parent_token[node])])
            node = int(self._parent[node])
        path.reverse()
        return path

    def _key_for_node(self, node):
        return self._key_from_path(self._node_path(node))

    def _iter_nodes_sorted(self, root, shallow):
        stack = [root]
        while stack:
            node = stack.pop()
            valued = bool(self._has_value[node])
            if valued:
                yield node
            if not (shallow and valued):
                children = self._children(node, sorted_children=True)
                stack.extend(child for _, child in reversed(children))

    def iteritems(self, prefix=_EMPTY, shallow=False):
        root = self._find_node(prefix) if prefix is not _EMPTY else 0
        nodes = (
            self._iter_nodes_sorted(root, shallow)
            if self._sorted
            else self._collect_nodes(root, shallow)
        )
        for node in nodes:
            node = int(node)
            yield self._key_for_node(node), self._values[node]

    def iterkeys(self, prefix=_EMPTY, shallow=False):
        for key, _ in self.iteritems(prefix, shallow):
            yield key

    def itervalues(self, prefix=_EMPTY, shallow=False):
        root = self._find_node(prefix) if prefix is not _EMPTY else 0
        nodes = (
            self._iter_nodes_sorted(root, shallow)
            if self._sorted
            else self._collect_nodes(root, shallow)
        )
        for node in nodes:
            yield self._values[int(node)]

    def items(self, prefix=_EMPTY, shallow=False):
        return list(self.iteritems(prefix, shallow))

    def keys(self, prefix=_EMPTY, shallow=False):
        return list(self.iterkeys(prefix, shallow))

    def values(self, prefix=_EMPTY, shallow=False):
        return list(self.itervalues(prefix, shallow))

    def __iter__(self):
        return self.iterkeys()

    def __len__(self):
        return self._size

    def __bool__(self):
        return bool(self._size)

    def has_node(self, key):
        try:
            node = self._find_node(key)
        except KeyError:
            return 0
        return self.HAS_VALUE * bool(self._has_value[node]) | self.HAS_SUBTRIE * (
            self._first[node] >= 0
        )

    def has_key(self, key):
        return bool(self.has_node(key) & self.HAS_VALUE)

    def has_subtrie(self, key):
        return bool(self.has_node(key) & self.HAS_SUBTRIE)

    class _NoneStep:
        key = value = None
        is_set = has_subtrie = False

        def __bool__(self):
            return False

        def get(self, default=None):
            return default

        def __getitem__(self, index):
            if index in (0, 1):
                return None
            raise IndexError("index out of range")

        def __repr__(self):
            return "(None Step)"

    class _Step:
        def __init__(self, trie, node):
            self._trie = trie
            self._node = node

        def __bool__(self):
            return True

        @property
        def key(self):
            return self._trie._key_for_node(self._node)

        @property
        def value(self):
            if not self._trie._has_value[self._node]:
                raise ShortKeyError(self.key)
            return self._trie._values[self._node]

        @value.setter
        def value(self, value):
            if value is _EMPTY:
                if self._trie._has_value[self._node]:
                    self._trie._has_value[self._node] = 0
                    self._trie._values[self._node] = _EMPTY
                    self._trie._size -= 1
                    self._trie._node_by_path.clear()
            else:
                self._trie._assign_node(self._node, value)

        @property
        def is_set(self):
            return bool(self._trie._has_value[self._node])

        @property
        def has_subtrie(self):
            return self._trie._first[self._node] >= 0

        def get(self, default=None):
            return self._trie._values[self._node] if self.is_set else default

        def set(self, value):
            self.value = value

        def setdefault(self, value):
            if not self.is_set:
                self.value = value
            return self.value

        def __getitem__(self, index):
            if index == 0:
                return self.key
            if index == 1:
                return self.value
            raise IndexError("index out of range")

        def __repr__(self):
            return f"({self.key!r}: {self.value!r})"

    _NONE_STEP = _NoneStep()

    def _trace(self, key):
        path = self._path(key)
        tokens = []
        complete_encoding = True
        for step in path:
            try:
                tokens.append(self._token_by_step[step])
            except KeyError:
                complete_encoding = False
                break
        encoded = np.asarray(tokens, dtype=np.int64)
        result = np.empty(len(tokens) + 1, dtype=np.int64)
        if not encoded.size:
            result[0] = 0
            return result[:1], complete_encoding and not path
        count = int(
            lib().mpg_trace(
                addr(self._hash_nodes),
                addr(self._hash_tokens),
                addr(self._hash_children),
                self._hash_nodes.size,
                addr(encoded),
                encoded.size,
                addr(result),
            )
        )
        matched_all = complete_encoding and count == len(path) + 1
        return result[:count], matched_all

    def walk_towards(self, key):
        nodes, complete = self._trace(key)
        for node in nodes:
            yield self._Step(self, int(node))
        if not complete:
            raise KeyError(key)

    def prefixes(self, key):
        try:
            for step in self.walk_towards(key):
                if step.is_set:
                    yield step
        except KeyError:
            return

    def shortest_prefix(self, key):
        return next(self.prefixes(key), self._NONE_STEP)

    def longest_prefix(self, key):
        result = self._NONE_STEP
        for result in self.prefixes(key):
            pass
        return result

    def compile_keys(self, keys):
        """Encode a reusable key batch for repeated bulk queries."""
        keys = list(keys)
        paths = [self._path(key) for key in keys]
        flat, offsets = self._encode_paths(paths, create=False)
        return KeyBatch(self, self._generation, keys, flat, offsets)

    def _pack_keys(self, keys):
        if isinstance(keys, KeyBatch):
            if keys._owner is not self or keys._generation != self._generation:
                raise ValueError("KeyBatch belongs to a different trie state")
            packed = keys.keys, keys.tokens, keys.offsets
            self._validate_batch(*packed)
            return packed
        batch = self.compile_keys(keys)
        return batch.keys, batch.tokens, batch.offsets

    @staticmethod
    def _validate_batch(keys, tokens, offsets):
        if not isinstance(keys, list):
            raise TypeError("KeyBatch keys must be a list")
        if not isinstance(tokens, np.ndarray) or tokens.dtype != np.int64:
            raise TypeError("KeyBatch tokens must be an int64 NumPy array")
        if not isinstance(offsets, np.ndarray) or offsets.dtype != np.int64:
            raise TypeError("KeyBatch offsets must be an int64 NumPy array")
        if tokens.ndim != 1 or not tokens.flags.c_contiguous:
            raise ValueError("KeyBatch tokens must be a contiguous 1-D array")
        if offsets.ndim != 1 or not offsets.flags.c_contiguous:
            raise ValueError("KeyBatch offsets must be a contiguous 1-D array")
        if len(offsets) != len(keys) + 1:
            raise ValueError("KeyBatch offset count does not match its keys")
        if offsets[0] != 0 or offsets[-1] != len(tokens):
            raise ValueError("KeyBatch offsets do not span its token buffer")
        if np.any(offsets < 0) or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("KeyBatch offsets must be non-negative and ordered")

    def bulk_get(self, keys, default=_EMPTY):
        """Return values for many keys with one Mojo traversal call."""
        if self._raw_keys_are_cache_keys and not isinstance(keys, KeyBatch):
            keys = list(keys)
            cache = self._node_by_path
            try:
                return [self._values[cache[key]] for key in keys]
            except (KeyError, TypeError):
                pass
        keys, flat, offsets = self._pack_keys(keys)
        if not keys:
            return []
        result = np.empty(len(keys), dtype=np.int64)
        ffi_flat = flat if flat.size else np.zeros(1, dtype=np.int64)
        lib().mpg_bulk_find(
            addr(self._hash_nodes),
            addr(self._hash_tokens),
            addr(self._hash_children),
            self._hash_nodes.size,
            addr(ffi_flat),
            addr(offsets),
            len(keys),
            addr(result),
        )
        if result.min(initial=0) >= 0 and self._has_value[result].all():
            return [self._values[int(node)] for node in result]
        values = []
        for key, node in zip(keys, result):
            node = int(node)
            if node >= 0 and self._has_value[node]:
                values.append(self._values[node])
            elif default is not _EMPTY:
                values.append(default)
            else:
                raise KeyError(key)
        return values

    def bulk_longest_prefix(self, keys):
        """Return longest-prefix Step objects for many keys in one Mojo call."""
        keys, flat, offsets = self._pack_keys(keys)
        if not keys:
            return []
        result = np.empty(len(keys), dtype=np.int64)
        ffi_flat = flat if flat.size else np.zeros(1, dtype=np.int64)
        lib().mpg_bulk_longest(
            addr(self._hash_nodes),
            addr(self._hash_tokens),
            addr(self._hash_children),
            self._hash_nodes.size,
            addr(self._has_value),
            addr(ffi_flat),
            addr(offsets),
            len(keys),
            addr(result),
        )
        return [
            self._NONE_STEP if node < 0 else self._Step(self, int(node))
            for node in result
        ]

    def _set_node_if_no_prefix(self, key):
        if self.shortest_prefix(key):
            return
        node = self._set_node(key, True)
        self._drop_children(node)

    def copy(self, _make_copy=lambda value: value):
        result = self.__class__()
        if isinstance(self, StringTrie):
            result = self.__class__(separator=self._separator)
        result._set_many([(key, _make_copy(value)) for key, value in self.items()])
        result.enable_sorting(self._sorted)
        return result

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(lambda value: _copy.deepcopy(value, memo))

    def merge(self, other, overwrite=False):
        if isinstance(other, StringTrie) and not isinstance(self, StringTrie):
            raise TypeError(
                f"{type(other).__name__} cannot be merged into a {type(self).__name__}"
            )
        pairs = []
        for node in other._collect_nodes():
            node = int(node)
            key = self._key_from_path(other._node_path(node))
            if overwrite or not self.has_key(key):
                pairs.append((key, other._values[node]))
        self._set_many(pairs)
        other.clear()

    def strictly_equals(self, other):
        if type(self) is not type(other):
            return False
        if isinstance(self, StringTrie) and self._separator != other._separator:
            return False
        return self == other

    def __eq__(self, other):
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        if len(self) != len(other):
            return False
        try:
            return all(key in other and other[key] == value for key, value in self.items())
        except (KeyError, TypeError):
            return False

    def __ne__(self, other):
        return not self == other

    def _str_items(self, fmt="%s: %s"):
        return ", ".join(fmt % item for item in self.iteritems())

    def __str__(self):
        return f"{type(self).__name__}({self._str_items()})"

    def __repr__(self):
        return f"{type(self).__name__}([{self._str_items('(%r, %r)')}])"

    def traverse(self, node_factory, prefix=_EMPTY):
        root = self._find_node(prefix) if prefix is not _EMPTY else 0

        def visit(node):
            path = tuple(self._node_path(node))
            raw_children = self._children(node)
            children = (
                _ChildrenIterator(lambda: (visit(child) for _, child in raw_children))
                if raw_children
                else ()
            )
            args = (self._key_from_path, path, children)
            if self._has_value[node]:
                return node_factory(*args, self._values[node])
            return node_factory(*args)

        return visit(root)

    traverse.uses_bool_convertible_children = True


class CharTrie(Trie):
    _raw_keys_are_cache_keys = True

    def _cache_key(self, key, path):
        try:
            hash(key)
        except TypeError:
            return path
        return key

    def _cache_key_without_path(self, key):
        try:
            hash(key)
        except TypeError:
            return None
        return key

    def _key_from_path(self, path):
        return "".join(path)


class StringTrie(Trie):
    _raw_keys_are_cache_keys = True

    def __init__(self, *args, **kwargs):
        separator = kwargs.pop("separator", "/")
        if not isinstance(separator, str):
            raise TypeError("separator must be a string")
        if not separator:
            raise ValueError("separator can not be empty")
        self._separator = separator
        super().__init__(*args, **kwargs)

    @classmethod
    def fromkeys(cls, keys, value=None, separator="/"):
        trie = cls(separator=separator)
        trie._set_many([(key, value) for key in keys])
        return trie

    def _path_from_key(self, key):
        return key.split(self._separator)

    def _cache_key(self, key, path):
        return key

    def _cache_key_without_path(self, key):
        return key

    def _key_from_path(self, path):
        return self._separator.join(path)

    def __str__(self):
        if not self:
            return f"{type(self).__name__}(separator={self._separator})"
        return (
            f"{type(self).__name__}({self._str_items()}, "
            f"separator={self._separator})"
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}([{self._str_items('(%r, %r)')}], "
            f"separator={self._separator!r})"
        )


class PrefixSet(collections.abc.MutableSet):
    def __init__(self, iterable=(), factory=Trie, **kwargs):
        self._trie = factory(**kwargs)
        for key in iterable:
            self.add(key)

    def __contains__(self, key):
        return bool(self._trie.shortest_prefix(key))

    def __iter__(self):
        return self._trie.iterkeys()

    def __len__(self):
        return len(self._trie)

    def add(self, value):
        self._trie._set_node_if_no_prefix(value)

    def iter(self, prefix=_EMPTY):
        if prefix is _EMPTY:
            return iter(self)
        if self._trie.has_node(prefix):
            return self._trie.iterkeys(prefix)
        if prefix in self:
            return (self._trie._key_from_path(self._trie._path_from_key(prefix)),)
        return ()

    def clear(self):
        self._trie.clear()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        result = self.__class__()
        result._trie = self._trie.copy()
        return result

    def __deepcopy__(self, memo):
        result = self.__class__()
        result._trie = _copy.deepcopy(self._trie, memo)
        return result

    def discard(self, value):
        raise NotImplementedError("Removing values from PrefixSet is not implemented.")

    def remove(self, value):
        raise NotImplementedError("Removing values from PrefixSet is not implemented.")

    def pop(self):
        raise NotImplementedError("Removing values from PrefixSet is not implemented.")


__all__ = [
    "Trie",
    "CharTrie",
    "StringTrie",
    "PrefixSet",
    "ShortKeyError",
    "KeyBatch",
]
