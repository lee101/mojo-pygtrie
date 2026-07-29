# mojo-pygtrie

`mojo-pygtrie` is a standalone Mojo port of the trie structures in
[`pygtrie`](https://github.com/mina86/pygtrie). Generic trie keys may contain
arbitrary hashable Python components, values may be any Python objects, and
structural traversal runs in one compiled Mojo shared library.

The Python package is imported as `mojopygtrie` so it can be parity-tested
beside the real `pygtrie` package:

```python
import mojopygtrie as pygtrie

routes = pygtrie.StringTrie(separator="/")
routes["api/v1/users"] = "users"
routes["api/v1/teams"] = "teams"

match = routes.longest_prefix("api/v1/users/42")
assert (match.key, match.value) == ("api/v1/users", "users")

batch = routes.compile_keys(["api/v1/users", "api/v1/teams"])
assert routes.bulk_get(batch) == ["users", "teams"]
```

## Upstream coverage

The covered subset mirrors pygtrie 2.5.0:

- `Trie`, `CharTrie`, and `StringTrie`, including the mutable mapping protocol;
- exact lookup, `has_node`, `has_key`, and `has_subtrie`;
- `walk_towards`, `prefixes`, `shortest_prefix`, and `longest_prefix`;
- prefix-limited and shallow iteration, sorting, and subtrie slices;
- insertion, update, `setdefault`, deletion, `pop`, `popitem`, and `clear`;
- `fromkeys`, shallow/deep copy, equality, `strictly_equals`, `merge`, and
  `traverse`;
- `PrefixSet`;
- additional `compile_keys`, `bulk_get`, and `bulk_longest_prefix` methods for
  workloads large enough to amortize the Python-to-Mojo boundary.

Tests compare these operations directly with the PyPI `pygtrie==2.5.0`
package, including randomized mutation sequences. The following upstream
behavior is not covered:

- `PrefixSet.discard`, `remove`, and `pop` raise `NotImplementedError`;
- pickles created by upstream pygtrie are not wire compatible, although native
  mojo-pygtrie pickle round-trips work;
- private implementation details such as `_Node`, `_Children`, and `_root` are
  not reproduced, so code that subclasses or inspects them is unsupported;
- simultaneous mutation of one trie from multiple threads is unsupported.

Deleted branches are detached immediately but their array slots are not
compacted. `clear()` releases that retained structural capacity, so an
extreme-churn service should clear or rebuild the trie periodically.

## Install and run from a source checkout

The repository pins the tested Mojo nightly and all Python dependencies:

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` produces `dist/libmojo-pygtrie.so`. The Python wrapper also
rebuilds it on first import from a source checkout when the library is missing
or older than the Mojo source. Set `MOJO_PYGTRIE_LIB` to the path of an
already-built shared library in another deployment. This repository does not
currently publish wheels or promise a system-wide `pip install`.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic, Python 3.13.14, Mojo
1.0.0b3.dev2026072406. Times are the best of three runs.

| workload | mojo-pygtrie | pygtrie 2.5.0 | result |
| --- | ---: | ---: | ---: |
| construct 100k-key StringTrie | 345.22 ms | 172.58 ms | 2.00x slower |
| 50k scalar exact lookups | 39.00 ms | 83.70 ms | 2.15x faster |
| 50k bulk exact, raw keys | 13.58 ms | 83.70 ms | 6.16x faster |
| 50k bulk exact, compiled keys | 18.33 ms | 83.70 ms | 4.57x faster |
| 50k longest-prefix, compiled keys | 50.02 ms | 226.10 ms | 4.52x faster |
| iterate 100k values | 32.65 ms | 45.19 ms | 1.38x faster |

Exact StringTrie and CharTrie lookups use an exact-key-to-node cache, while
uncached structural queries and compiled batches use Mojo traversal. Batch
packing builds one contiguous NumPy token buffer instead of allocating an array
per key. Construction-time integer buffer fills and growth copies use native
SIMD with scalar remainder handling.

There is no GPU path. Trie traversal performs a few dependent hash-table probes
per component and moves far more data than it computes, so its arithmetic
intensity is well below the level where device transfer and launch costs are
justified. The bulk traversal is also left serial: individual queries are
independent, but each does too little memory-bound work to amortize CPU task
launch overhead.

## How it works

Python interns each hashable key component to an integer token and retains
values in a Python list, so values may still be any Python object. A Python
dictionary caches terminal StringTrie and CharTrie keys to their node indices;
mutations that remove values or detach nodes invalidate it. Mojo sees only caller-owned
contiguous buffers:

- each node stores its first and last child edge, parent, and parent token;
- each edge stores a token, child node, and next-sibling index;
- an open-addressed hash table maps `(node, token)` to child node;
- a byte array marks nodes that own values.

`src/trie.mojo` is the single compilation unit. Its exported functions use
`@export("name")` and `abi("C")`; NumPy buffer addresses cross ctypes as
64-bit integers and are reconstructed as
`UnsafePointer[..., AnyOrigin[mut=True]]` inside Mojo. Mojo performs no heap
allocation and owns no Python memory. Batched insertion, exact lookup, prefix
tracing, longest-prefix lookup, and depth-first value collection each cross
the FFI boundary once.

## License

MIT
