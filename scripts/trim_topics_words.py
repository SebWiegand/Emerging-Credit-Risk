"""
Trim topic words so output matches what topic_labeling_llm.py expects.

Input (default):  outputs_textual_factors/topics_words.csv
Output (default): outputs_textual_factors/topics_words_trimmed.csv

Output columns are ALWAYS:
- topic (int)
- top_words_pretty (comma-separated string)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import ast
import re

import numpy as np
import pandas as pd


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Pick first matching column (case-insensitive) from candidates."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(
        f"Could not find any of columns {candidates} in input. Available columns: {list(df.columns)}"
    )


def _clean_numpy_wrappers(s: str) -> str:
    """
    Convert things like np.float64(0.123) / numpy.float32(0.5) / np.int64(7) -> 0.123 / 0.5 / 7
    so ast.literal_eval can parse the dict/list.
    """
    s = re.sub(r"np\.(?:float\d+|float_|float)\(([^)]+)\)", r"\1", s)
    s = re.sub(r"np\.(?:int\d+|int_|int)\(([^)]+)\)", r"\1", s)
    s = re.sub(r"numpy\.(?:float\d+|float_|float)\(([^)]+)\)", r"\1", s)
    s = re.sub(r"numpy\.(?:int\d+|int_|int)\(([^)]+)\)", r"\1", s)
    return s


def _parse_topic_distribution(x: Any) -> list[tuple[str, float]]:
    """
    Parse topic_distribution into list of (word, weight).
    Supports:
    - dict {word: weight}
    - list of pairs [(word, weight), ...]
    - stringified python literal dict/list (including np.float64 wrappers)
    - fallback "word:0.1, word2:0.05" style
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []

    if isinstance(x, dict):
        out = []
        for k, v in x.items():
            try:
                out.append((str(k), float(v)))
            except Exception:
                pass
        return out

    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    out.append((str(item[0]), float(item[1])))
                except Exception:
                    pass
        return out

    s = str(x).strip()
    if not s:
        return []
    s = _clean_numpy_wrappers(s)

    # Try python-literal parsing (handles single quotes etc.)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            out = []
            for k, v in obj.items():
                try:
                    out.append((str(k), float(v)))
                except Exception:
                    pass
            return out
        if isinstance(obj, (list, tuple)):
            out = []
            for item in obj:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    try:
                        out.append((str(item[0]), float(item[1])))
                    except Exception:
                        pass
            if out:
                return out
    except Exception:
        pass

    # Fallback: parse simple "word:weight, word2:weight2" strings
    s2 = _clean_numpy_wrappers(str(x).strip())
    pairs: list[tuple[str, float]] = []
    for part in [p.strip() for p in s2.replace(";", ",").split(",") if p.strip()]:
        if ":" not in part and "=" not in part:
            continue
        sep = ":" if ":" in part else "="
        w, wt = part.split(sep, 1)
        w = w.strip().strip("\"'")
        try:
            pairs.append((w, float(wt.strip())))
        except Exception:
            continue
    return pairs


def trim_topics_words(in_path: Path, out_path: Path, top_n: int = 25) -> pd.DataFrame:
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    df = pd.read_csv(in_path)
    cols_lower = {c.lower() for c in df.columns}

    # ---------- Case A: compact schema: topic + topic_distribution ----------
    if "topic" in cols_lower and "topic_distribution" in cols_lower:
        df = df.rename(
            columns={
                _pick_col(df, ["topic"]): "topic",
                _pick_col(df, ["topic_distribution"]): "topic_distribution",
            }
        )

        rows: list[dict[str, Any]] = []
        for _, r in df.iterrows():
            topic = r.get("topic")
            dist = _parse_topic_distribution(r.get("topic_distribution"))
            for word, weight in dist:
                rows.append({"topic": topic, "word": word, "weight": weight})

        long_df = pd.DataFrame(rows)
        if long_df.empty:
            example = df["topic_distribution"].dropna().astype(str).head(1).tolist()
            example_str = example[0][:500] if example else "<no non-empty values>"
            raise ValueError(
                "Parsed topic_distribution but got 0 (word, weight) pairs. "
                "Format is unexpected. "
                f"Example value (first 500 chars): {example_str}"
            )

        long_df["abs_weight"] = pd.to_numeric(long_df["weight"], errors="coerce").abs()
        long_df["topic"] = pd.to_numeric(long_df["topic"], errors="coerce")
        long_df = long_df.dropna(subset=["topic", "word", "abs_weight"]).copy()

        top = (
            long_df.sort_values(["topic", "abs_weight"], ascending=[True, False])
            .groupby("topic", as_index=False, sort=False)
            .head(top_n)
            .drop(columns=["abs_weight"])
        )

        summary = (
            top.groupby("topic", as_index=False)
            .agg(top_words_pretty=("word", lambda s: ", ".join(map(str, s))))
        )

    # ---------- Case B: long schema: cluster/topic + word + weight ----------
    else:
        cluster_col = next(
            (c for c in df.columns if c.lower() in {"cluster", "cluster_id", "clusterid", "cluster_idx", "group", "bucket"}),
            None,
        )
        topic_col = next(
            (c for c in df.columns if c.lower() in {"topic", "topic_id", "topicid", "topic_idx", "component"}),
            None,
        )
        word_col = next(
            (c for c in df.columns if c.lower() in {"word", "token", "term", "vocab", "feature"}),
            None,
        )
        weight_col = next(
            (c for c in df.columns if c.lower() in {"weight", "loading", "score", "coef", "coefficient", "value"}),
            None,
        )

        if cluster_col is None and topic_col is None:
            raise KeyError(f"Could not find topic identifiers in input. Available columns: {list(df.columns)}")
        if word_col is None or weight_col is None:
            raise KeyError(f"Could not find word/weight columns in input. Available columns: {list(df.columns)}")

        work = df.copy()
        # topic_labeling_llm.py merges on "topic" vs "cluster" in importances,
        # so we prefer cluster if available.
        if cluster_col is not None:
            work = work.rename(columns={cluster_col: "topic"})
        else:
            work = work.rename(columns={topic_col: "topic"})

        work = work.rename(columns={word_col: "word", weight_col: "weight"})
        work["abs_weight"] = pd.to_numeric(work["weight"], errors="coerce").abs()
        work["topic"] = pd.to_numeric(work["topic"], errors="coerce")
        work = work.dropna(subset=["topic", "word", "abs_weight"]).copy()

        top = (
            work.sort_values(["topic", "abs_weight"], ascending=[True, False])
            .groupby("topic", as_index=False, sort=False)
            .head(top_n)
            .drop(columns=["abs_weight"])
        )

        summary = (
            top.groupby("topic", as_index=False)
            .agg(top_words_pretty=("word", lambda s: ", ".join(map(str, s))))
        )

    # Final standardization
    summary["topic"] = pd.to_numeric(summary["topic"], errors="coerce").astype("Int64")
    summary = summary.dropna(subset=["topic"]).copy()
    summary["topic"] = summary["topic"].astype(int)

    trimmed = summary[["topic", "top_words_pretty"]].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed.to_csv(out_path, index=False)
    return trimmed


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    in_path = repo_root / "outputs_textual_factors" / "topics_words.csv"
    out_path = repo_root / "outputs_textual_factors" / "topics_words_trimmed.csv"

    trimmed = trim_topics_words(in_path=in_path, out_path=out_path, top_n=25)
    print(f"Read:  {in_path}")
    print(f"Wrote: {out_path}")
    print(f"Rows:  {len(trimmed):,}")


if __name__ == "__main__":
    main()