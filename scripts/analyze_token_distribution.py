#!/usr/bin/env python3
"""
CLI for directory-level token distribution analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maestro.data.token_distribution import DEFAULT_TEXT_EXTENSIONS, analyze_token_distribution, write_token_distribution_report


def _parse_extensions(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze token distributions for text files in a directory.")
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing text files.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Tokenizer model id or local tokenizer directory.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=None,
        help=f"Comma-separated extension list. Default: {','.join(DEFAULT_TEXT_EXTENSIONS)}",
    )
    parser.add_argument("--workers", type=int, default=None, help="Worker processes (default: cpu_count).")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Files per multiprocessing task.")
    parser.add_argument("--histogram-bins", type=int, default=40, help="Number of histogram bins.")
    parser.add_argument("--top-files", type=int, default=30, help="Number of largest files-by-token to include.")
    parser.add_argument("--top-tokens", type=int, default=200, help="Top token ids for TF/DF tables.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path.")
    parser.add_argument("--no-recursive", action="store_true", help="Only scan files directly under the directory.")
    parser.add_argument("--follow-symlinks", action="store_true", help="Follow symlinks while scanning.")
    parser.add_argument("--include-token-frequency", action="store_true", help="Compute top token TF/DF tables.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    out_path = args.output
    if out_path is None:
        out_path = str(Path(args.directory).expanduser().resolve() / "token_distribution_report.json")

    report = analyze_token_distribution(
        directory=args.directory,
        tokenizer_source=args.tokenizer,
        recursive=not args.no_recursive,
        extensions=_parse_extensions(args.extensions),
        workers=args.workers,
        chunk_size=args.chunk_size,
        histogram_bins=args.histogram_bins,
        top_k_files=args.top_files,
        include_token_frequency=args.include_token_frequency,
        top_k_tokens=args.top_tokens,
        max_files=args.max_files,
        follow_symlinks=args.follow_symlinks,
        show_progress=not args.no_progress,
    )
    write_token_distribution_report(report, out_path)

    md = report["metadata"]
    agg = report["aggregate"]
    print(f"Saved report: {out_path}")
    print(
        "processed={processed}/{total} failed={failed} total_tokens={tokens} "
        "avg_tokens_per_file={avg:.2f} elapsed={elapsed:.2f}s".format(
            processed=md["processed_files"],
            total=md["total_files_scanned"],
            failed=md["failed_files"],
            tokens=agg.get("total_tokens", 0),
            avg=agg.get("avg_tokens_per_file", 0.0),
            elapsed=md["elapsed_seconds"],
        )
    )


if __name__ == "__main__":
    main()
