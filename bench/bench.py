"""Benchmarks against pygtrie 2.5.0 on identical keys and operations."""

import gc
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import pygtrie as upstream  # noqa: E402

import mojopygtrie as mojo  # noqa: E402


def best_time(function, repetitions=3):
    best = float("inf")
    result = None
    for _ in range(repetitions):
        gc.collect()
        started = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - started)
    return best, result


def machine():
    cpu = platform.processor()
    if (not cpu or cpu.lower() in ("x86_64", "amd64")) and os.path.exists(
        "/proc/cpuinfo"
    ):
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    mojo_version = subprocess.run(
        ["mojo", "--version"], capture_output=True, check=True, text=True
    ).stdout.strip()
    return (
        f"{cpu or 'unknown CPU'}; {platform.system()} {platform.release()}; "
        f"Python {platform.python_version()}; {mojo_version}; "
        f"pygtrie {version('pygtrie')}"
    )


def relative(mojo_seconds, reference_seconds):
    ratio = reference_seconds / mojo_seconds
    if ratio >= 1:
        return f"{ratio:.2f}x faster"
    return f"{1 / ratio:.2f}x slower"


def main():
    count = 100_000
    pairs = [(f"api/v1/users/{index:06d}", index) for index in range(count)]

    mojo_build, mojo_trie = best_time(lambda: mojo.StringTrie(pairs))
    ref_build, ref_trie = best_time(lambda: upstream.StringTrie(pairs))
    assert mojo_trie.items() == ref_trie.items()

    query_keys = [key for key, _ in pairs[::2]]
    prefix_queries = [key + "/profile" for key in query_keys]
    compiled_exact = mojo_trie.compile_keys(query_keys)
    compiled_prefix = mojo_trie.compile_keys(prefix_queries)

    rows = [("construct 100k-key StringTrie", mojo_build, ref_build)]

    mojo_time, mojo_result = best_time(
        lambda: sum(mojo_trie[key] for key in query_keys)
    )
    ref_time, ref_result = best_time(lambda: sum(ref_trie[key] for key in query_keys))
    assert mojo_result == ref_result
    rows.append(("50k scalar exact lookups", mojo_time, ref_time))

    mojo_time, mojo_result = best_time(lambda: sum(mojo_trie.bulk_get(query_keys)))
    assert mojo_result == ref_result
    rows.append(("50k bulk exact, raw keys", mojo_time, ref_time))

    mojo_time, mojo_result = best_time(
        lambda: sum(mojo_trie.bulk_get(compiled_exact))
    )
    assert mojo_result == ref_result
    rows.append(("50k bulk exact, compiled keys", mojo_time, ref_time))

    mojo_time, mojo_result = best_time(
        lambda: sum(step.value for step in mojo_trie.bulk_longest_prefix(compiled_prefix))
    )
    ref_time, ref_result = best_time(
        lambda: sum(ref_trie.longest_prefix(key).value for key in prefix_queries)
    )
    assert mojo_result == ref_result
    rows.append(("50k longest-prefix, compiled keys", mojo_time, ref_time))

    mojo_time, mojo_result = best_time(lambda: sum(mojo_trie.values()))
    ref_time, ref_result = best_time(lambda: sum(ref_trie.values()))
    assert mojo_result == ref_result
    rows.append(("iterate 100k values", mojo_time, ref_time))

    print(f"Machine: {machine()}")
    print()
    print("| workload | mojo-pygtrie | pygtrie 2.5.0 | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_seconds, ref_seconds in rows:
        print(
            f"| {name} | {mojo_seconds * 1e3:.2f} ms | "
            f"{ref_seconds * 1e3:.2f} ms | {relative(mojo_seconds, ref_seconds)} |"
        )


if __name__ == "__main__":
    main()
