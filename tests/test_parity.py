import copy
import pickle
import random

import numpy as np
import pytest
import pygtrie as upstream

import mojopygtrie as mojo
from mojopygtrie import _i64_filled
from mojopygtrie._lib import addr


CLASSES = [
    (mojo.Trie, upstream.Trie),
    (mojo.CharTrie, upstream.CharTrie),
    (mojo.StringTrie, upstream.StringTrie),
]


def keys_for(cls):
    if cls is mojo.Trie or cls is upstream.Trie:
        return [("a",), ("a", "b"), ("a", "b", "c"), ("x",)]
    if cls is mojo.CharTrie or cls is upstream.CharTrie:
        return ["a", "ab", "abc", "x"]
    return ["a", "a/b", "a/b/c", "x"]


@pytest.mark.parametrize("ours,reference", CLASSES)
def test_mapping_construction_and_lookup(ours, reference):
    keys = keys_for(ours)
    data = dict(zip(keys, [None, 2, 3, 4]))
    got, expected = ours(data), reference(data)
    assert len(got) == len(expected)
    assert got.items() == expected.items()
    assert [got[key] for key in keys] == [expected[key] for key in keys]
    assert list(got) == list(expected)
    assert bool(got) == bool(expected)


@pytest.mark.parametrize("ours,reference", CLASSES)
def test_update_setdefault_get_and_fromkeys(ours, reference):
    keys = keys_for(ours)
    got, expected = ours(), reference()
    got.update([(keys[0], 1), (keys[1], 2)])
    expected.update([(keys[0], 1), (keys[1], 2)])
    assert got.setdefault(keys[0], 99) == expected.setdefault(keys[0], 99)
    assert got.setdefault(keys[2], 3) == expected.setdefault(keys[2], 3)
    assert got.get(keys[3], "missing") == expected.get(keys[3], "missing")
    assert got.items() == expected.items()
    assert ours.fromkeys(keys, 7).items() == reference.fromkeys(keys, 7).items()


@pytest.mark.parametrize("ours,reference", CLASSES)
def test_prefix_queries_and_steps(ours, reference):
    keys = keys_for(ours)
    got, expected = ours(), reference()
    for index, key in enumerate(keys[:3]):
        got[key] = expected[key] = index
    probe = keys[2]
    got_steps, expected_steps = list(got.prefixes(probe)), list(expected.prefixes(probe))
    assert [(s.key, s.value) for s in got_steps] == [
        (s.key, s.value) for s in expected_steps
    ]
    assert got.shortest_prefix(probe).key == expected.shortest_prefix(probe).key
    assert got.longest_prefix(probe).key == expected.longest_prefix(probe).key
    got_steps[0].value = 20
    expected_steps[0].value = 20
    assert got.items() == expected.items()
    assert not got.longest_prefix(keys[3])


@pytest.mark.parametrize("ours,reference", CLASSES)
def test_has_node_flags(ours, reference):
    keys = keys_for(ours)
    got, expected = ours(), reference()
    got[keys[1]] = expected[keys[1]] = 1
    got[keys[2]] = expected[keys[2]] = 2
    for key in keys:
        assert got.has_node(key) == expected.has_node(key)
        assert got.has_key(key) == expected.has_key(key)
        assert got.has_subtrie(key) == expected.has_subtrie(key)


@pytest.mark.parametrize("ours,reference", CLASSES)
def test_subtrie_iteration_shallow_and_slices(ours, reference):
    keys = keys_for(ours)
    got = ours(dict(zip(keys, range(4))))
    expected = reference(dict(zip(keys, range(4))))
    prefix = keys[0]
    assert got.items(prefix) == expected.items(prefix)
    assert got.items(prefix, shallow=True) == expected.items(prefix, shallow=True)
    assert list(got[prefix:]) == list(expected[prefix:])
    got[prefix:] = expected[prefix:] = 42
    assert got.items() == expected.items()


@pytest.mark.parametrize("ours,reference", CLASSES)
def test_delete_pop_and_pruning(ours, reference):
    keys = keys_for(ours)
    got = ours(dict(zip(keys, range(4))))
    expected = reference(dict(zip(keys, range(4))))
    assert got.pop(keys[1]) == expected.pop(keys[1])
    assert got.pop(keys[1], "d") == expected.pop(keys[1], "d")
    del got[keys[0]]
    del expected[keys[0]]
    assert got.items() == expected.items()
    del got[keys[2]:]
    del expected[keys[2]:]
    assert got.items() == expected.items()
    assert got.popitem() == expected.popitem()


def test_short_key_and_invalid_slice_errors_match():
    got, expected = mojo.StringTrie(), upstream.StringTrie()
    got["a/b"] = expected["a/b"] = 1
    with pytest.raises(mojo.ShortKeyError):
        _ = got["a"]
    with pytest.raises(upstream.ShortKeyError):
        _ = expected["a"]
    with pytest.raises(TypeError):
        _ = got["a":"b"]


@pytest.mark.parametrize("ours,reference", CLASSES)
def test_sorted_iteration_matches_upstream(ours, reference):
    keys = list(reversed(keys_for(ours)))
    got, expected = ours(), reference()
    for index, key in enumerate(keys):
        got[key] = expected[key] = index
    got.enable_sorting()
    expected.enable_sorting()
    assert got.items() == expected.items()
    got.enable_sorting(False)
    expected.enable_sorting(False)
    assert got.items() == expected.items()


def test_string_separator_and_empty_key_parity():
    got = mojo.StringTrie(separator=".")
    expected = upstream.StringTrie(separator=".")
    for key, value in [("", 0), (".admin", 1), (".admin.images", 2)]:
        got[key] = expected[key] = value
    assert got.items() == expected.items()
    assert got.longest_prefix(".admin.images.logo").key == expected.longest_prefix(
        ".admin.images.logo"
    ).key
    with pytest.raises(ValueError):
        mojo.StringTrie(separator="")
    with pytest.raises(TypeError):
        mojo.StringTrie(separator=1)


def test_generic_hashable_components_and_char_output():
    got = mojo.Trie()
    expected = upstream.Trie()
    keys = [(1, "x", None), (1, "x", False), (2,)]
    for index, key in enumerate(keys):
        got[key] = expected[key] = index
    assert got.items() == expected.items()
    chars = mojo.CharTrie({"word": 1, "worm": 2})
    chars[list("wide")] = 3
    assert all(isinstance(key, str) for key in chars.keys())


def test_copy_deepcopy_pickle_and_equality():
    original = mojo.StringTrie({"a/b": [1], "a/c": [2]})
    shallow = copy.copy(original)
    deep = copy.deepcopy(original)
    shallow["a/b"].append(3)
    assert original["a/b"] == [1, 3]
    assert deep["a/b"] == [1]
    restored = pickle.loads(pickle.dumps(original))
    assert restored == original
    assert original == {"a/b": [1, 3], "a/c": [2]}
    assert original.strictly_equals(shallow)
    assert not original.strictly_equals(mojo.StringTrie(original, separator="."))


def test_merge_moves_other_and_honours_overwrite():
    got = mojo.StringTrie({"a/b": 1, "x": 9})
    expected = upstream.StringTrie({"a/b": 1, "x": 9})
    got_other = mojo.StringTrie({"a/b": 2, "c/d": 3})
    expected_other = upstream.StringTrie({"a/b": 2, "c/d": 3})
    got.merge(got_other)
    expected.merge(expected_other)
    assert got.items() == expected.items()
    assert not got_other and not expected_other
    replacement = mojo.StringTrie({"a/b": 7})
    got.merge(replacement, overwrite=True)
    assert got["a/b"] == 7


def test_walk_towards_missing_yields_existing_steps_then_raises():
    got, expected = mojo.CharTrie({"abc": 1}), upstream.CharTrie({"abc": 1})

    def consume(trie):
        iterator = trie.walk_towards("abz")
        seen = []
        with pytest.raises(KeyError):
            while True:
                seen.append(next(iterator).key)
        return seen

    assert consume(got) == consume(expected)


def test_traverse_parity():
    got = mojo.StringTrie({"a": 1, "a/b": 2, "x/y": 3})
    expected = upstream.StringTrie({"a": 1, "a/b": 2, "x/y": 3})

    def factory(path_conv, path, children, *value):
        return (path_conv(path), value[0] if value else None, list(children))

    assert got.traverse(factory) == expected.traverse(factory)


def test_prefix_set_parity():
    got = mojo.PrefixSet(factory=mojo.CharTrie)
    expected = upstream.PrefixSet(factory=upstream.CharTrie)
    for key in ("foobar", "foobaz", "bar", "foo"):
        got.add(key)
        expected.add(key)
    assert list(got) == list(expected)
    assert len(got) == len(expected)
    for key in ("foo", "fooz", "barista", "qux"):
        assert (key in got) == (key in expected)
    assert tuple(got.iter("foobar")) == tuple(expected.iter("foobar"))
    with pytest.raises(NotImplementedError):
        got.discard("foo")


def test_bulk_exact_and_longest_prefix_match_upstream():
    keys = [f"api/v{i // 100}/resource/{i}" for i in range(1000)]
    values = list(range(len(keys)))
    got = mojo.StringTrie(zip(keys, values))
    expected = upstream.StringTrie(zip(keys, values))
    queries = keys[::3] + ["api/v1/resource/missing", "other"]
    assert got.bulk_get(queries, None) == [expected.get(key) for key in queries]
    got_prefixes = got.bulk_longest_prefix(key + "/tail" for key in queries)
    expected_prefixes = [expected.longest_prefix(key + "/tail") for key in queries]
    assert [(s.key, s.get()) for s in got_prefixes] == [
        (s.key, s.get()) for s in expected_prefixes
    ]
    batch = got.compile_keys(queries)
    assert got.bulk_get(batch, None) == [expected.get(key) for key in queries]
    got.clear()
    with pytest.raises(ValueError):
        got.bulk_get(batch)


def test_empty_native_operations_and_empty_key_batches():
    trie = mojo.StringTrie({"": 1})
    assert trie[""] == 1
    assert trie.bulk_get([]) == []
    assert trie.bulk_longest_prefix([]) == []
    batch = trie.compile_keys(["", "missing"])
    assert trie.bulk_get(batch, None) == [1, None]
    assert [step.get() for step in trie.bulk_longest_prefix(batch)] == [1, None]


def test_corrupt_key_batches_are_rejected_before_native_call():
    trie = mojo.StringTrie({"a/b": 1})
    batch = trie.compile_keys(["a/b"])
    batch.offsets[-1] = len(batch.tokens) + 1
    with pytest.raises(ValueError, match="span"):
        trie.bulk_get(batch)

    batch = trie.compile_keys(["a/b"])
    batch.tokens = batch.tokens.astype(np.int32)
    with pytest.raises(TypeError, match="int64"):
        trie.bulk_longest_prefix(batch)


def test_native_address_rejects_incompatible_numpy_buffers():
    assert addr(np.zeros(1, dtype=np.int64))
    assert addr(np.zeros(1, dtype=np.uint8))
    with pytest.raises(TypeError):
        addr(np.zeros(1, dtype=np.int32))
    with pytest.raises(ValueError):
        addr(np.zeros(4, dtype=np.int64)[::2])


def test_simd_fill_and_grow_scalar_tails():
    original = _i64_filled(19, -7)
    assert original.tolist() == [-7] * 19
    original[:] = range(19)
    grown = mojo.Trie._grow(original, 23, -11)
    assert grown.tolist() == list(range(19)) + [-11] * 4


def test_bulk_construction_across_insertion_chunk_boundary():
    count = mojo.Trie._INSERT_CHUNK + 17
    pairs = [(f"shared/prefix/{index}", index) for index in range(count)]
    trie = mojo.StringTrie(pairs)
    assert len(trie) == count
    assert trie[pairs[0][0]] == 0
    assert trie[pairs[-1][0]] == count - 1
    assert trie.items() == upstream.StringTrie(pairs).items()


def test_bulk_string_construction_parent_reuse_and_duplicates():
    pairs = [
        ("a::b::one", 1),
        ("a::b::two", 2),
        ("x::one", 3),
        ("a::b::one", 4),
        ("x::two", 5),
    ]
    got = mojo.StringTrie(pairs, separator="::")
    expected = upstream.StringTrie(pairs, separator="::")
    assert got.items() == expected.items()
    assert len(got) == len(expected)


def test_cached_exact_lookup_is_invalidated_by_pruning():
    trie = mojo.StringTrie({"a/b": 1, "a/c": 2})
    assert trie.bulk_get(["a/b", "a/c"]) == [1, 2]
    assert trie["a/b"] == 1
    del trie["a":]
    with pytest.raises(KeyError):
        _ = trie["a/b"]
    trie["a/b"] = 3
    assert trie["a/b"] == 3
    trie["a"] = 4
    assert trie.pop("a") == 4
    assert trie.bulk_get(["a"], None) == [None]


def test_randomised_mutation_parity():
    rng = random.Random(0)
    keys = [f"{a}/{b}/{c}" for a in "abc" for b in range(8) for c in range(4)]
    got, expected = mojo.StringTrie(), upstream.StringTrie()
    for operation in range(1000):
        key = rng.choice(keys)
        if rng.random() < 0.65:
            value = (operation, None if operation % 7 == 0 else key)
            got[key] = expected[key] = value
        else:
            assert got.pop(key, None) == expected.pop(key, None)
        assert len(got) == len(expected)
    got.enable_sorting()
    expected.enable_sorting()
    assert got.items() == expected.items()
