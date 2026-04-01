"""
Utilities for multiprocessing token distribution analysis over text file trees.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from heapq import heappush, heappushpop
from pathlib import Path
from typing import Any, Iterable, Iterator

from transformers import AutoTokenizer

DEFAULT_TEXT_EXTENSIONS = (
    ".txt",
    ".py",
    ".lua",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".tsv",
    ".xml",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".sh",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".java",
    ".rs",
    ".go",
    ".sql",
)

_TOKENIZER: Any = None
_INCLUDE_TOKEN_FREQUENCY = False


@dataclass
class _FileRecord:
    path: str
    extension: str
    tokens: int
    chars: int
    lines: int
    bytes_size: int


def _normalize_extensions(extensions: Iterable[str] | None) -> set[str]:
    if extensions is None:
        return set(DEFAULT_TEXT_EXTENSIONS)
    out: set[str] = set()
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        out.add(ext)
    return out


def _iter_text_files(
    directory: Path,
    extensions: set[str],
    recursive: bool,
    follow_symlinks: bool,
) -> Iterator[Path]:
    if recursive:
        for root, _, files in os.walk(directory, followlinks=follow_symlinks):
            for name in files:
                path = Path(root) / name
                if path.suffix.lower() in extensions:
                    yield path
        return

    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def _chunked(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _init_worker(tokenizer_source: str, trust_remote_code: bool, include_token_frequency: bool) -> None:
    global _TOKENIZER, _INCLUDE_TOKEN_FREQUENCY
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code, use_fast=True)
    _INCLUDE_TOKEN_FREQUENCY = include_token_frequency


def _process_chunk(paths: list[str]) -> dict[str, Any]:
    ext_totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "files": 0,
            "total_tokens": 0,
            "total_chars": 0,
            "total_lines": 0,
            "total_bytes": 0,
            "token_counts": [],
        }
    )
    token_tf: Counter[int] = Counter()
    token_df: Counter[int] = Counter()
    file_records: list[_FileRecord] = []
    errors: list[dict[str, str]] = []

    total_tokens = 0
    total_chars = 0
    total_lines = 0
    total_bytes = 0
    processed_files = 0

    for raw_path in paths:
        path = Path(raw_path)
        ext = path.suffix.lower()

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            try:
                ids = _TOKENIZER.encode(text, add_special_tokens=False, truncation=False, verbose=False)
            except TypeError:
                ids = _TOKENIZER.encode(text, add_special_tokens=False)
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
            continue

        token_count = len(ids)
        char_count = len(text)
        line_count = text.count("\n") + (1 if text else 0)
        byte_count = path.stat().st_size

        file_records.append(
            _FileRecord(
                path=str(path),
                extension=ext,
                tokens=token_count,
                chars=char_count,
                lines=line_count,
                bytes_size=byte_count,
            )
        )

        total_tokens += token_count
        total_chars += char_count
        total_lines += line_count
        total_bytes += byte_count
        processed_files += 1

        ext_row = ext_totals[ext]
        ext_row["files"] += 1
        ext_row["total_tokens"] += token_count
        ext_row["total_chars"] += char_count
        ext_row["total_lines"] += line_count
        ext_row["total_bytes"] += byte_count
        ext_row["token_counts"].append(token_count)

        if _INCLUDE_TOKEN_FREQUENCY:
            file_counter = Counter(ids)
            token_tf.update(file_counter)
            token_df.update(file_counter.keys())

    return {
        "processed_files": processed_files,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "total_lines": total_lines,
        "total_bytes": total_bytes,
        "file_records": [record.__dict__ for record in file_records],
        "token_tf": dict(token_tf),
        "token_df": dict(token_df),
        "extension_totals": dict(ext_totals),
        "errors": errors,
    }


def _percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])

    idx = (len(sorted_values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(sorted_values[lo])
    frac = idx - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def _distribution_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "std": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    sorted_values = sorted(values)
    n = len(sorted_values)
    total = sum(sorted_values)
    mean = total / n
    variance = sum((x - mean) ** 2 for x in sorted_values) / n

    return {
        "count": n,
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": mean,
        "std": math.sqrt(variance),
        "p01": _percentile(sorted_values, 0.01),
        "p05": _percentile(sorted_values, 0.05),
        "p10": _percentile(sorted_values, 0.10),
        "p25": _percentile(sorted_values, 0.25),
        "p50": _percentile(sorted_values, 0.50),
        "p75": _percentile(sorted_values, 0.75),
        "p90": _percentile(sorted_values, 0.90),
        "p95": _percentile(sorted_values, 0.95),
        "p99": _percentile(sorted_values, 0.99),
    }


def _histogram(values: list[int], bins: int) -> list[dict[str, float | int]]:
    if not values or bins <= 0:
        return []

    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        return [{"bin_start": float(vmin), "bin_end": float(vmax), "file_count": len(values)}]

    step = (vmax - vmin) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        idx = int((value - vmin) / step)
        if idx == bins:
            idx = bins - 1
        counts[idx] += 1

    out: list[dict[str, float | int]] = []
    for i, count in enumerate(counts):
        start = vmin + (i * step)
        end = vmin + ((i + 1) * step)
        out.append({"bin_start": float(start), "bin_end": float(end), "file_count": count})
    return out


def analyze_token_distribution(
    directory: str | Path,
    tokenizer_source: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    *,
    recursive: bool = True,
    extensions: Iterable[str] | None = None,
    workers: int | None = None,
    chunk_size: int = 2048,
    histogram_bins: int = 40,
    top_k_files: int = 30,
    include_token_frequency: bool = False,
    top_k_tokens: int = 200,
    trust_remote_code: bool = True,
    max_files: int | None = None,
    follow_symlinks: bool = False,
    show_progress: bool = True,
    max_error_records: int = 200,
) -> dict[str, Any]:
    """
    Analyze token distributions for text files in a directory tree with multiprocessing.
    """
    started_at = time.time()
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Directory not found: {root}")

    normalized_exts = _normalize_extensions(extensions)
    file_paths = [str(p) for p in _iter_text_files(root, normalized_exts, recursive=recursive, follow_symlinks=follow_symlinks)]
    if max_files is not None:
        file_paths = file_paths[:max_files]

    total_files = len(file_paths)
    if total_files == 0:
        return {
            "metadata": {
                "directory": str(root),
                "tokenizer_source": tokenizer_source,
                "recursive": recursive,
                "extensions": sorted(normalized_exts),
                "workers": 0,
                "chunk_size": chunk_size,
                "total_files_scanned": 0,
                "processed_files": 0,
                "failed_files": 0,
                "elapsed_seconds": round(time.time() - started_at, 3),
            },
            "aggregate": {},
            "by_extension": {},
            "top_files_by_tokens": [],
            "file_token_count_histogram": [],
            "token_frequency_top": [],
            "token_document_frequency_top": [],
            "errors": [],
        }

    resolved_workers = workers or multiprocessing.cpu_count()
    resolved_workers = max(1, min(resolved_workers, multiprocessing.cpu_count()))
    chunk_size = max(1, chunk_size)

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None

    top_k_files = max(0, int(top_k_files))
    top_k_tokens = max(0, int(top_k_tokens))
    total_chunks = (total_files + chunk_size - 1) // chunk_size

    file_token_counts: list[int] = []
    extension_totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "files": 0,
            "total_tokens": 0,
            "total_chars": 0,
            "total_lines": 0,
            "total_bytes": 0,
            "token_counts": [],
        }
    )
    token_tf: Counter[int] = Counter()
    token_df: Counter[int] = Counter()
    top_heap: list[tuple[int, str, dict[str, Any]]] = []
    errors: list[dict[str, str]] = []

    total_tokens = 0
    total_chars = 0
    total_lines = 0
    total_bytes = 0
    processed_files = 0

    with multiprocessing.Pool(
        processes=resolved_workers,
        initializer=_init_worker,
        initargs=(tokenizer_source, trust_remote_code, include_token_frequency),
    ) as pool:
        result_iter = pool.imap_unordered(_process_chunk, _chunked(file_paths, chunk_size))
        if show_progress and tqdm is not None:
            result_iter = tqdm(result_iter, total=total_chunks, desc="Token analysis")

        for result in result_iter:
            processed_files += int(result["processed_files"])
            total_tokens += int(result["total_tokens"])
            total_chars += int(result["total_chars"])
            total_lines += int(result["total_lines"])
            total_bytes += int(result["total_bytes"])

            if len(errors) < max_error_records:
                remaining = max_error_records - len(errors)
                errors.extend(result["errors"][:remaining])

            for row in result["file_records"]:
                file_token_counts.append(int(row["tokens"]))
                if top_k_files > 0:
                    row_key = {
                        "path": row["path"],
                        "tokens": row["tokens"],
                        "chars": row["chars"],
                        "lines": row["lines"],
                        "bytes": row["bytes_size"],
                    }
                    item = (int(row["tokens"]), str(row["path"]), row_key)
                    if len(top_heap) < top_k_files:
                        heappush(top_heap, item)
                    elif item > top_heap[0]:
                        heappushpop(top_heap, item)

            for ext, ext_row in result["extension_totals"].items():
                dst = extension_totals[ext]
                dst["files"] += int(ext_row["files"])
                dst["total_tokens"] += int(ext_row["total_tokens"])
                dst["total_chars"] += int(ext_row["total_chars"])
                dst["total_lines"] += int(ext_row["total_lines"])
                dst["total_bytes"] += int(ext_row["total_bytes"])
                dst["token_counts"].extend(int(x) for x in ext_row["token_counts"])

            if include_token_frequency:
                token_tf.update({int(k): int(v) for k, v in result["token_tf"].items()})
                token_df.update({int(k): int(v) for k, v in result["token_df"].items()})

    failed_files = total_files - processed_files
    elapsed_seconds = time.time() - started_at

    top_files = [x[2] for x in sorted(top_heap, key=lambda x: x[0], reverse=True)]
    file_token_stats = _distribution_summary(file_token_counts)

    by_extension: dict[str, Any] = {}
    for ext, row in sorted(extension_totals.items(), key=lambda kv: kv[1]["total_tokens"], reverse=True):
        by_extension[ext] = {
            "files": row["files"],
            "total_tokens": row["total_tokens"],
            "total_chars": row["total_chars"],
            "total_lines": row["total_lines"],
            "total_bytes": row["total_bytes"],
            "avg_tokens_per_file": (row["total_tokens"] / row["files"]) if row["files"] else 0.0,
            "token_count_distribution": _distribution_summary(row["token_counts"]),
        }

    token_frequency_top: list[dict[str, Any]] = []
    token_document_frequency_top: list[dict[str, Any]] = []
    if include_token_frequency and top_k_tokens > 0:
        decode_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code, use_fast=True)
        for token_id, count in token_tf.most_common(top_k_tokens):
            token_text = decode_tokenizer.convert_ids_to_tokens(int(token_id))
            token_frequency_top.append(
                {
                    "token_id": int(token_id),
                    "token": token_text,
                    "count": int(count),
                    "fraction_of_all_tokens": (count / total_tokens) if total_tokens else 0.0,
                }
            )

        for token_id, count in token_df.most_common(top_k_tokens):
            token_text = decode_tokenizer.convert_ids_to_tokens(int(token_id))
            token_document_frequency_top.append(
                {
                    "token_id": int(token_id),
                    "token": token_text,
                    "file_count": int(count),
                    "fraction_of_files": (count / processed_files) if processed_files else 0.0,
                }
            )

    report = {
        "metadata": {
            "directory": str(root),
            "tokenizer_source": tokenizer_source,
            "recursive": recursive,
            "extensions": sorted(normalized_exts),
            "workers": resolved_workers,
            "chunk_size": chunk_size,
            "total_files_scanned": total_files,
            "processed_files": processed_files,
            "failed_files": failed_files,
            "analysis_started_unix": started_at,
            "elapsed_seconds": round(elapsed_seconds, 3),
        },
        "aggregate": {
            "total_tokens": total_tokens,
            "total_chars": total_chars,
            "total_lines": total_lines,
            "total_bytes": total_bytes,
            "avg_tokens_per_file": (total_tokens / processed_files) if processed_files else 0.0,
            "avg_chars_per_file": (total_chars / processed_files) if processed_files else 0.0,
            "avg_lines_per_file": (total_lines / processed_files) if processed_files else 0.0,
            "tokens_per_char": (total_tokens / total_chars) if total_chars else 0.0,
            "files_per_second": (processed_files / elapsed_seconds) if elapsed_seconds else 0.0,
            "tokens_per_second": (total_tokens / elapsed_seconds) if elapsed_seconds else 0.0,
            "file_token_count_distribution": file_token_stats,
        },
        "by_extension": by_extension,
        "top_files_by_tokens": top_files,
        "file_token_count_histogram": _histogram(file_token_counts, histogram_bins),
        "token_frequency_top": token_frequency_top,
        "token_document_frequency_top": token_document_frequency_top,
        "errors": errors,
    }
    return report


def write_token_distribution_report(report: dict[str, Any], output_path: str | Path) -> None:
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
