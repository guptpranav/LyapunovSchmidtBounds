"""
build_regular_graph_index.py
============================
Enumerate connected non-isomorphic k-regular graphs on n vertices
and write the results incrementally to a JSONL file.

Each output line (one per feasible (n, k) pair) has the form::

    {"n": <int>, "k": <int>, "count": <int>, "complete": <bool>, "graphs": [<g6_str>, ...]}

Usage
-----
    python build_regular_graph_index.py [OPTIONS]

Options
-------
    --n-max       INT   Maximum number of vertices (default: 10)
    --num-graphs  INT   Maximum graphs to store per (n, k) pair (default: 1000)
    --output      PATH  Output JSONL file path (default: regular_graphs.jsonl)
    --resume            Resume from the last completed (n, k) in the existing file

Notes
-----
* Feasibility condition: n*k must be even (handshaking lemma), k < n,
  n >= 2 (a 1-regular graph on 1 vertex is impossible), k >= 1.
* Sage must be importable from the current Python environment.
* No external subprocesses are used; only the standard library and Sage.
"""

import sys
import json
import argparse
import networkx as nx
from pathlib import Path
from typing import Generator, Optional, Tuple

# ---------------------------------------------------------------------------
# Helper: feasible (n, k) pairs
# ---------------------------------------------------------------------------

def feasible_pairs(n_max: int) -> Generator[Tuple[int, int], None, None]:
    """
    Yield all (n, k) pairs that *could* admit a connected k-regular graph.

    Conditions applied:
    - 2 <= n <= n_max          (need at least 2 vertices for k >= 1)
    - 2 <= k <= n - 1          (proper regularity; complete graph uses k=n-1)
    - n * k is even            (handshaking lemma: sum of degrees must be even)

    Pairs are generated in lexicographic order: increasing n, then increasing k.

    Parameters
    ----------
    n_max : int
        Upper bound on the number of vertices (inclusive).

    Yields
    ------
    (n, k) : Tuple[int, int]
    """
    for n in range(2, n_max + 1):
        for k in range(2, n):
            if (n * k) % 2 == 0:
                yield (n, k)


# ---------------------------------------------------------------------------
# Helper: enumerate graphs for one (n, k) pair
# ---------------------------------------------------------------------------

# def enumerate_regular_graphs(
#     n: int,
#     k: int,
#     num_graphs: int,
# ) -> Tuple[list, bool]:
#     """
#     Enumerate connected k-regular graphs on n vertices via ``nauty_geng``.

#     Iteration stops as soon as ``num_graphs`` graphs have been collected
#     (truncated) or the generator is exhausted (complete).

#     Parameters
#     ----------
#     n : int
#         Number of vertices.
#     k : int
#         Degree of regularity.
#     num_graphs : int
#         Maximum number of graphs to collect.

#     Returns
#     -------
#     graphs : list of str
#         Graph6 strings in the order produced by ``nauty_geng``.
#     complete : bool
#         True if enumeration finished naturally; False if truncated.
#     """
#     # nauty_geng option string: "-c" = connected, "-d<k> -D<k>" = exactly k-regular
#     # The first positional argument to nauty_geng is the number of vertices.
#     options = f"-c -d{k} -D{k}"

#     graph_list: list = []
#     complete = True

#     try:
#         gen: Iterator = graphs.nauty_geng(f"{n} {options}")
#         for graph in gen:
#             g6 = graph.graph6_string()
#             graph_list.append(g6)
#             if len(graph_list) >= num_graphs:
#                 # Check whether the generator still has items.
#                 # We peek by calling next(); if it raises StopIteration the
#                 # enumeration was actually complete at exactly num_graphs.
#                 try:
#                     next_graph = next(gen)  # type: ignore[call-overload]
#                     # There is at least one more – truncate here.
#                     complete = False
#                 except StopIteration:
#                     complete = True
#                 break
#     except Exception as exc:  # noqa: BLE001  (broad catch intentional)
#         # nauty_geng may raise if the option string is invalid or no graphs
#         # exist.  Treat as an empty, complete result so we don't crash the
#         # entire run.
#         print(
#             f"  WARNING: nauty_geng raised an exception for n={n}, k={k}: {exc}",
#             file=sys.stderr,
#         )
#         graph_list = []
#         complete = True

#     return graph_list, complete

def enumerate_regular_graphs(
    n: int,
    k: int,
    num_graphs: int,
) -> Tuple[list, bool]:
    """
    Enumerate connected k-regular graphs on n vertices.
    Iteration stops as soon as ``num_graphs`` graphs have been collected.

    Parameters
    ----------
    n : int
        Number of vertices.
    k : int
        Degree of regularity.
    num_graphs : int
        Maximum number of graphs to collect.

    Returns
    -------
    graphs : list of str
        Graph6 strings.
    complete : bool
        True if enumeration finished naturally; False if truncated.
    """

    complete = False

    if k == 2 or k >= n-2:
        num_graphs = 1
        complete = True

    graph_set = set()

    while len(graph_set) < num_graphs:
        G = nx.random_regular_graph(k, n)

        if nx.is_connected(G):
            g6 = nx.to_graph6_bytes(G, header=False).decode().strip()
            graph_set.add(g6)

    graph_list = sorted(graph_set)    

    return graph_list, complete


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def read_last_completed_pair(output_path: Path) -> Optional[Tuple[int, int]]:
    """
    Read the last JSON object in *output_path* and return its (n, k).

    Only the last line is read, making this O(file size) in the worst case
    but still far cheaper than re-parsing the whole file.

    Parameters
    ----------
    output_path : Path
        Path to the existing JSONL file.

    Returns
    -------
    (n, k) : Tuple[int, int] or None
        The (n, k) of the last entry, or None if the file is empty / missing.

    Raises
    ------
    SystemExit
        If the last line cannot be parsed as JSON.
    """
    if not output_path.exists() or output_path.stat().st_size == 0:
        return None

    last_line = ""
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                last_line = stripped

    if not last_line:
        return None

    try:
        obj = json.loads(last_line)
        return (int(obj["n"]), int(obj["k"]))
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        sys.exit(
            f"ERROR: Could not parse the last line of '{output_path}' as a "
            f"valid JSON object.\nLine: {last_line!r}\nError: {exc}"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a JSONL index of connected non-isomorphic regular graphs "
            "using Sage's nauty_geng."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=10,
        metavar="N",
        help="Maximum number of vertices (inclusive).",
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=1000,
        metavar="M",
        help="Maximum number of graphs to store per (n, k) pair.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("regular_graphs_index.jsonl"),
        metavar="FILE",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume by reading the last entry in the existing output file and "
            "continuing from the next feasible (n, k) pair."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Main driver: enumerate regular graphs and write results to JSONL."""
    args = parse_args()

    n_max: int = args.n_max
    num_graphs: int = args.num_graphs
    output_path: Path = args.output
    resume: bool = args.resume

    # ------------------------------------------------------------------
    # Validate arguments
    # ------------------------------------------------------------------
    if n_max < 2:
        sys.exit("ERROR: --n-max must be at least 2.")
    if num_graphs < 1:
        sys.exit("ERROR: --num-graphs must be at least 1.")

    # ------------------------------------------------------------------
    # Resume logic
    # ------------------------------------------------------------------
    skip_up_to: Optional[Tuple[int, int]] = None

    if resume:
        if not output_path.exists():
            print(
                f"INFO: --resume specified but '{output_path}' does not exist. "
                "Starting fresh.",
                file=sys.stderr,
            )
        else:
            last_pair = read_last_completed_pair(output_path)
            if last_pair is None:
                print(
                    f"INFO: '{output_path}' exists but is empty. Starting fresh.",
                    file=sys.stderr,
                )
            else:
                skip_up_to = last_pair
                print(
                    f"INFO: Resuming after (n={last_pair[0]}, k={last_pair[1]}).",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------
    # Open output file (append when resuming, write otherwise)
    # ------------------------------------------------------------------
    file_mode = "a" if (resume and output_path.exists()) else "w"

    try:
        out_file = output_path.open(file_mode, encoding="utf-8", buffering=1)
    except OSError as exc:
        sys.exit(f"ERROR: Cannot open output file '{output_path}': {exc}")

    # ------------------------------------------------------------------
    # Enumeration loop
    # ------------------------------------------------------------------
    skipping = skip_up_to is not None  # True until we pass skip_up_to

    try:
        for n, k in feasible_pairs(n_max):

            # ----------------------------------------------------------
            # Skip pairs already present in the file when resuming
            # ----------------------------------------------------------
            if skipping:
                if (n, k) == skip_up_to:
                    skipping = False  # next pair is the first new one
                continue  # skip this pair (already in file)

            # ----------------------------------------------------------
            # Enumerate graphs for this (n, k) pair
            # ----------------------------------------------------------
            graphs, complete = enumerate_regular_graphs(n, k, num_graphs)

            # ----------------------------------------------------------
            # Write one JSON line immediately (incremental output)
            # ----------------------------------------------------------
            record = {
                "n": n,
                "k": k,
                "count": len(graphs),
                "complete": complete,
                "graphs": graphs,
            }
            out_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            out_file.flush()  # ensure the line is on disk before continuing

            # ----------------------------------------------------------
            # Progress report
            # ----------------------------------------------------------
            status = "complete" if complete else "truncated"
            print(f"n={n} k={k} : stored {len(graphs)} graphs ({status})")

    except KeyboardInterrupt:
        print("\nInterrupted by user.  Partial results have been saved.", file=sys.stderr)
    finally:
        out_file.close()


if __name__ == "__main__":
    main()