#!/usr/bin/env python3
"""
core025_skip_engine_app_v3_1__2026-03-25.py

Core 025 Skip Engine v3.1
- Deep negative-trait discovery
- Stacked skip buckets
- Sidebar controls restored
- Current scoring table
- Historical scoring table
- Play/skip plan summaries

This app is intentionally skip-focused.
It does NOT choose members and does NOT rank straights.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None


CORE025_SET = {"0025", "0225", "0255"}
DIGITS = list(range(10))
MIRROR_PAIRS = {(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)}


# ---------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------

def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: Dict[str, int] = {}
    cols: List[str] = []
    for col in df.columns:
        name = str(col)
        if name not in seen:
            seen[name] = 0
            cols.append(name)
        else:
            seen[name] += 1
            cols.append(f"{name}__dup{seen[name]}")
    out = df.copy()
    out.columns = cols
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    cols = list(df.columns)
    nmap = {_norm(c): c for c in cols}
    for cand in candidates:
        key = _norm(cand)
        if key in nmap:
            return nmap[key]
    for cand in candidates:
        key = _norm(cand)
        for k, c in nmap.items():
            if key and key in k:
                return c
    if required:
        raise KeyError(f"Required column not found. Tried {list(candidates)}. Available columns: {cols}")
    return None


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return dedupe_columns(df).to_csv(index=False).encode("utf-8")


def safe_display_df(df: pd.DataFrame, rows: int) -> pd.DataFrame:
    return dedupe_columns(df).head(int(rows)).copy()


def percentile_rank_series(s: pd.Series) -> pd.Series:
    if len(s) == 0:
        return s
    return s.rank(method="average", pct=True)


def has_streamlit_context() -> bool:
    if st is None:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


# ---------------------------------------------------------
# Loading and normalization
# ---------------------------------------------------------

def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".txt") or name.endswith(".tsv"):
        data = uploaded_file.getvalue()
        try:
            return pd.read_csv(io.BytesIO(data), sep="\t", header=None)
        except Exception:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", header=None)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError(f"Unsupported uploaded input type: {uploaded_file.name}")


def normalize_result_to_4digits(result_text: str) -> Optional[str]:
    if pd.isna(result_text):
        return None
    digits = re.findall(r"\d", str(result_text))
    if len(digits) < 4:
        return None
    return "".join(digits[:4])


def core025_member(result4: str) -> Optional[str]:
    if result4 is None:
        return None
    sorted4 = "".join(sorted(result4))
    return sorted4 if sorted4 in CORE025_SET else None


def prepare_history(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = dedupe_columns(df_raw.copy())

    if len(df.columns) == 4:
        c0, c1, c2, c3 = list(df.columns)
        df = df.rename(columns={c0: "date", c1: "jurisdiction", c2: "game", c3: "result_raw"})
    else:
        date_col = find_col(df, ["date"], required=True)
        juris_col = find_col(df, ["jurisdiction", "state", "province"], required=True)
        game_col = find_col(df, ["game", "stream"], required=True)
        result_col = find_col(df, ["result", "winning result", "draw result"], required=True)
        df = df.rename(columns={
            date_col: "date",
            juris_col: "jurisdiction",
            game_col: "game",
            result_col: "result_raw",
        })

    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["result4"] = df["result_raw"].apply(normalize_result_to_4digits)
    df["member"] = df["result4"].apply(core025_member)
    df["is_core025_hit"] = df["member"].notna().astype(int)
    df["stream_id"] = df["jurisdiction"].astype(str).str.strip() + " | " + df["game"].astype(str).str.strip()

    df = df.dropna(subset=["result4"]).copy().reset_index(drop=True)
    df["file_order"] = np.arange(len(df))
    return dedupe_columns(df)


# ---------------------------------------------------------
# Transition events
# ---------------------------------------------------------

def build_transition_events(history_df: pd.DataFrame) -> pd.DataFrame:
    # Input files are typically reverse chronological; this sorts chronologically within stream.
    sort_df = history_df.sort_values(["stream_id", "date_dt", "file_order"], ascending=[True, True, False]).copy()

    rows: List[Dict[str, object]] = []

    for stream_id, g in sort_df.groupby("stream_id", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue

        past_hit_positions: List[int] = []

        for i in range(1, len(g)):
            prev_row = g.iloc[i - 1]
            cur_row = g.iloc[i]

            last_hit_before_prev = past_hit_positions[-1] if len(past_hit_positions) > 0 else None
            current_gap_before_event = (i - 1 - last_hit_before_prev) if last_hit_before_prev is not None else i

            last50 = g.iloc[max(0, i - 50):i]
            recent_50_hit_rate = float(last50["is_core025_hit"].mean()) if len(last50) else 0.0

            rows.append({
                "stream_id": stream_id,
                "jurisdiction": cur_row["jurisdiction"],
                "game": cur_row["game"],
                "event_date": cur_row["date_dt"],
                "seed": prev_row["result4"],
                "next_result4": cur_row["result4"],
                "next_member": cur_row["member"] if pd.notna(cur_row["member"]) else "",
                "next_is_core025_hit": int(cur_row["is_core025_hit"]),
                "stream_event_index": int(i),
                "current_gap_before_event": int(current_gap_before_event),
                "recent_50_hit_rate_before_event": recent_50_hit_rate,
            })

            if int(cur_row["is_core025_hit"]) == 1:
                past_hit_positions.append(i)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError("No usable transitions could be created from the uploaded history.")
    return dedupe_columns(out)


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------

def digit_list(seed: str) -> List[int]:
    return [int(ch) for ch in str(seed)]


def feature_dict(seed: str) -> Dict[str, object]:
    d = digit_list(seed)
    cnt = Counter(d)
    s = sum(d)
    spread = max(d) - min(d)
    even = sum(x % 2 == 0 for x in d)
    odd = 4 - even
    high = sum(x >= 5 for x in d)
    low = 4 - high
    unique = len(cnt)

    consec_links = 0
    unique_sorted = sorted(set(d))
    for a, b in zip(unique_sorted[:-1], unique_sorted[1:]):
        if b - a == 1:
            consec_links += 1

    mirrorpair_cnt = sum(1 for a, b in MIRROR_PAIRS if a in cnt and b in cnt)

    out: Dict[str, object] = {
        "sum": s,
        "spread": spread,
        "even": even,
        "odd": odd,
        "high": high,
        "low": low,
        "unique": unique,
        "pair": int(unique < 4),
        "trip_or_more": int(max(cnt.values()) >= 3),
        "quad": int(max(cnt.values()) >= 4),
        "sum_mod3": s % 3,
        "sum_mod4": s % 4,
        "sum_mod5": s % 5,
        "pos1": d[0],
        "pos2": d[1],
        "pos3": d[2],
        "pos4": d[3],
        "outer_sum": d[0] + d[3],
        "inner_sum": d[1] + d[2],
        "outer_inner_gap": abs((d[0] + d[3]) - (d[1] + d[2])),
        "consec_links": consec_links,
        "mirrorpair_cnt": mirrorpair_cnt,
        "sorted_seed": "".join(map(str, sorted(d))),
        "parity_pattern": "".join("E" if x % 2 == 0 else "O" for x in d),
        "highlow_pattern": "".join("H" if x >= 5 else "L" for x in d),
    }

    out["cnt_0_3"] = int(sum(0 <= x <= 3 for x in d))
    out["cnt_4_6"] = int(sum(4 <= x <= 6 for x in d))
    out["cnt_7_9"] = int(sum(7 <= x <= 9 for x in d))

    for k in DIGITS:
        out[f"has{k}"] = int(k in cnt)
        out[f"cnt{k}"] = int(cnt.get(k, 0))

    return out


def build_feature_table(transitions_df: pd.DataFrame) -> pd.DataFrame:
    feats = [feature_dict(seed) for seed in transitions_df["seed"].astype(str)]
    feat_df = pd.DataFrame(feats)
    return dedupe_columns(pd.concat([transitions_df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1))


# ---------------------------------------------------------
# Trait mining
# ---------------------------------------------------------

def mine_negative_traits(df: pd.DataFrame, min_support: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    base_rate = float(df["next_is_core025_hit"].mean())

    candidate_cols = [
        "sum", "spread", "even", "odd", "high", "low", "unique", "pair", "trip_or_more",
        "quad", "sum_mod3", "sum_mod4", "sum_mod5", "pos1", "pos2", "pos3", "pos4",
        "outer_sum", "inner_sum", "outer_inner_gap", "consec_links", "mirrorpair_cnt",
        "cnt_0_3", "cnt_4_6", "cnt_7_9", "sorted_seed", "parity_pattern", "highlow_pattern"
    ] + [f"has{k}" for k in DIGITS] + [f"cnt{k}" for k in DIGITS]

    for col in candidate_cols:
        vals = df[col].dropna().unique().tolist()
        vals = sorted(vals)
        for val in vals:
            mask = df[col] == val
            support = int(mask.sum())
            if support < int(min_support):
                continue
            hit_rate = float(df.loc[mask, "next_is_core025_hit"].mean())
            rows.append({
                "trait": f"{col}={val}",
                "support": support,
                "support_pct": support / len(df),
                "hit_rate": hit_rate,
                "gain_vs_base": base_rate - hit_rate,
                "zero_hit_trait": int(hit_rate == 0.0),
            })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["zero_hit_trait", "hit_rate", "support"],
            ascending=[False, True, False]
        ).reset_index(drop=True)
    return dedupe_columns(out)


def eval_single_trait(df: pd.DataFrame, trait: str) -> pd.Series:
    col, raw_val = trait.split("=", 1)
    series = df[col]
    if pd.api.types.is_numeric_dtype(series):
        try:
            val = int(raw_val)
        except Exception:
            try:
                val = float(raw_val)
            except Exception:
                val = raw_val
    else:
        val = raw_val
    return series == val


def build_stacked_skip_buckets(
    df: pd.DataFrame,
    negative_traits: pd.DataFrame,
    bucket_min_support: int,
    bucket_top_k: int,
    bucket_max_depth: int,
    max_hit_rate_allowed: float,
) -> pd.DataFrame:
    target = df["next_is_core025_hit"].astype(int)
    base_rate = float(target.mean())

    candidates = negative_traits[
        (negative_traits["support"] >= int(bucket_min_support)) &
        (negative_traits["hit_rate"] <= float(max_hit_rate_allowed))
    ].head(int(bucket_top_k)).copy()

    rows: List[Dict[str, object]] = []
    seen = set()

    candidate_traits = candidates["trait"].tolist()

    for start_trait in candidate_traits:
        current_mask = eval_single_trait(df, start_trait)
        chosen = [start_trait]

        for _ in range(int(bucket_max_depth)):
            support = int(current_mask.sum())
            if support < int(bucket_min_support):
                break

            hit_rate = float(df.loc[current_mask, "next_is_core025_hit"].mean()) if support else 0.0
            key = tuple(chosen)
            if key not in seen:
                seen.add(key)
                rows.append({
                    "bucket_id": 0,
                    "depth": len(chosen),
                    "bucket_traits": " & ".join(chosen),
                    "support": support,
                    "support_pct": support / len(df),
                    "hit_rate": hit_rate,
                    "gain_vs_base": base_rate - hit_rate,
                    "zero_hit_bucket": int(hit_rate == 0.0),
                })

            best_next = None
            best_next_rate = hit_rate
            best_next_support = support
            best_next_mask = None

            for nxt in candidate_traits:
                if nxt in chosen:
                    continue
                nxt_mask = current_mask & eval_single_trait(df, nxt)
                nxt_support = int(nxt_mask.sum())
                if nxt_support < int(bucket_min_support):
                    continue
                nxt_rate = float(df.loc[nxt_mask, "next_is_core025_hit"].mean())
                if (nxt_rate < best_next_rate) or (np.isclose(nxt_rate, best_next_rate) and nxt_support > best_next_support):
                    best_next = nxt
                    best_next_rate = nxt_rate
                    best_next_support = nxt_support
                    best_next_mask = nxt_mask

            if best_next is None:
                break

            chosen.append(best_next)
            current_mask = best_next_mask

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["zero_hit_bucket", "hit_rate", "support", "depth"],
            ascending=[False, True, False, True]
        ).reset_index(drop=True)
        out["bucket_id"] = np.arange(1, len(out) + 1)
    return dedupe_columns(out)


def bucket_mask_from_string(df: pd.DataFrame, bucket_traits: str) -> pd.Series:
    parts = [p.strip() for p in str(bucket_traits).split("&") if p.strip()]
    if len(parts) == 0:
        return pd.Series([False] * len(df), index=df.index)
    mask = pd.Series([True] * len(df), index=df.index)
    for p in parts:
        mask &= eval_single_trait(df, p)
    return mask


# ---------------------------------------------------------
# Scoring
# ---------------------------------------------------------

def build_historical_scoring_table(
    feat_df: pd.DataFrame,
    skip_buckets: pd.DataFrame,
    top_skip_buckets_to_apply: int,
    skip_score_threshold: float,
) -> pd.DataFrame:
    work = feat_df.copy()
    selected = skip_buckets.head(int(top_skip_buckets_to_apply)).copy() if len(skip_buckets) else pd.DataFrame()

    fire_counts: List[int] = []
    fire_ids: List[str] = []

    for idx in work.index:
        row_df = work.loc[[idx]]
        ids: List[int] = []
        for _, b in selected.iterrows():
            if bool(bucket_mask_from_string(row_df, b["bucket_traits"]).iloc[0]):
                ids.append(int(b["bucket_id"]))
        fire_counts.append(len(ids))
        fire_ids.append(",".join(map(str, ids)))

    work["skip_bucket_fire_count"] = fire_counts
    work["fired_skip_bucket_ids"] = fire_ids

    # Stronger stream-level negative context
    work["stream_negative_pct"] = percentile_rank_series(1 - work.groupby("stream_id")["next_is_core025_hit"].transform("mean"))
    work["recent50_negative_pct"] = percentile_rank_series(1 - work["recent_50_hit_rate_before_event"].fillna(0))
    work["bucket_fire_pct"] = percentile_rank_series(work["skip_bucket_fire_count"].fillna(0))

    work["skip_score"] = (
        0.35 * work["stream_negative_pct"].fillna(0) +
        0.25 * work["recent50_negative_pct"].fillna(0) +
        0.40 * work["bucket_fire_pct"].fillna(0)
    ).clip(lower=0, upper=1)

    work["skip_class"] = np.where(work["skip_score"] >= float(skip_score_threshold), "SKIP", "PLAY")
    return dedupe_columns(work)


def summarize_scoring(scored_df: pd.DataFrame) -> pd.DataFrame:
    total_events = int(len(scored_df))
    total_hits = int(scored_df["next_is_core025_hit"].sum())

    rows = []
    for cls in ["SKIP", "PLAY"]:
        sub = scored_df[scored_df["skip_class"] == cls].copy()
        events = int(len(sub))
        hits = int(sub["next_is_core025_hit"].sum()) if len(sub) else 0
        rows.append({
            "skip_class": cls,
            "events": events,
            "events_pct": events / total_events if total_events else 0.0,
            "core025_hits": hits,
            "hit_rate": hits / events if events else 0.0,
            "hit_share_of_all_hits": hits / total_hits if total_hits else 0.0,
            "avg_skip_score": float(sub["skip_score"].mean()) if len(sub) else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_play_plans(scored_df: pd.DataFrame) -> pd.DataFrame:
    total_events = int(len(scored_df))
    total_hits = int(scored_df["next_is_core025_hit"].sum())

    rows = []

    plans = {
        "Skip only SKIP": scored_df["skip_class"] != "SKIP",
        "Play everything": pd.Series([True] * len(scored_df), index=scored_df.index),
    }

    for plan_name, play_mask in plans.items():
        played = scored_df[play_mask].copy()
        skipped = scored_df[~play_mask].copy()

        plays = int(len(played))
        hits_kept = int(played["next_is_core025_hit"].sum()) if len(played) else 0
        hits_skipped = int(skipped["next_is_core025_hit"].sum()) if len(skipped) else 0

        rows.append({
            "plan": plan_name,
            "plays": plays,
            "plays_pct_of_all_events": plays / total_events if total_events else 0.0,
            "plays_saved": total_events - plays,
            "plays_saved_pct": (total_events - plays) / total_events if total_events else 0.0,
            "core025_hits_kept": hits_kept,
            "core025_hits_skipped": hits_skipped,
            "hit_retention_pct": hits_kept / total_hits if total_hits else 0.0,
            "hit_rate_on_played_events": hits_kept / plays if plays else 0.0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Current scoring
# ---------------------------------------------------------

def compute_stream_profiles(transitions_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for stream_id, g in transitions_df.groupby("stream_id", sort=False):
        g = g.sort_values(["event_date", "stream_event_index"]).reset_index(drop=True)
        hits = int(g["next_is_core025_hit"].sum())
        events = int(len(g))
        rows.append({
            "stream_id": stream_id,
            "jurisdiction": g.iloc[-1]["jurisdiction"],
            "game": g.iloc[-1]["game"],
            "events": events,
            "core025_hits": hits,
            "core025_hit_rate": hits / events if events else 0.0,
            "recent_50_core025_rate": float(g.tail(50)["next_is_core025_hit"].mean()) if len(g) else 0.0,
        })
    out = pd.DataFrame(rows)
    return dedupe_columns(out)


def get_current_seed_rows(main_history: pd.DataFrame, last24_history: Optional[pd.DataFrame]) -> pd.DataFrame:
    source = last24_history if last24_history is not None and len(last24_history) else main_history
    source = source.sort_values(["stream_id", "date_dt", "file_order"], ascending=[True, True, False]).copy()
    latest = source.groupby("stream_id", sort=False).tail(1).copy()
    latest = latest[["stream_id", "jurisdiction", "game", "date_dt", "result4"]].rename(columns={
        "date_dt": "seed_date",
        "result4": "seed",
    })
    return dedupe_columns(latest.reset_index(drop=True))


def build_current_skip_table(
    current_seeds: pd.DataFrame,
    stream_profiles: pd.DataFrame,
    skip_buckets: pd.DataFrame,
    top_skip_buckets_to_apply: int,
    skip_score_threshold: float,
) -> pd.DataFrame:
    merged = current_seeds.merge(stream_profiles, on=["stream_id", "jurisdiction", "game"], how="left")
    feats = [feature_dict(seed) for seed in merged["seed"].astype(str)]
    feat_df = pd.DataFrame(feats)
    work = dedupe_columns(pd.concat([merged.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1))

    selected = skip_buckets.head(int(top_skip_buckets_to_apply)).copy() if len(skip_buckets) else pd.DataFrame()

    fire_counts: List[int] = []
    fire_ids: List[str] = []
    for idx in work.index:
        row_df = work.loc[[idx]]
        ids: List[int] = []
        for _, b in selected.iterrows():
            if bool(bucket_mask_from_string(row_df, b["bucket_traits"]).iloc[0]):
                ids.append(int(b["bucket_id"]))
        fire_counts.append(len(ids))
        fire_ids.append(",".join(map(str, ids)))

    work["skip_bucket_fire_count"] = fire_counts
    work["fired_skip_bucket_ids"] = fire_ids
    work["stream_negative_pct"] = percentile_rank_series(1 - work["core025_hit_rate"].fillna(0))
    work["recent50_negative_pct"] = percentile_rank_series(1 - work["recent_50_core025_rate"].fillna(0))
    work["bucket_fire_pct"] = percentile_rank_series(work["skip_bucket_fire_count"].fillna(0))

    work["skip_score"] = (
        0.35 * work["stream_negative_pct"].fillna(0) +
        0.25 * work["recent50_negative_pct"].fillna(0) +
        0.40 * work["bucket_fire_pct"].fillna(0)
    ).clip(lower=0, upper=1)

    work["skip_class"] = np.where(work["skip_score"] >= float(skip_score_threshold), "SKIP", "PLAY")

    keep_cols = [
        "stream_id", "jurisdiction", "game", "seed_date", "seed",
        "events", "core025_hits", "core025_hit_rate", "recent_50_core025_rate",
        "skip_bucket_fire_count", "fired_skip_bucket_ids", "skip_score", "skip_class",
    ]
    out = work[keep_cols].copy()
    out = out.sort_values(["skip_score", "skip_bucket_fire_count", "core025_hit_rate"], ascending=[False, False, True]).reset_index(drop=True)
    return dedupe_columns(out)


# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------

def build_summary_text(
    transitions_df: pd.DataFrame,
    negative_traits_df: pd.DataFrame,
    skip_buckets_df: pd.DataFrame,
    class_summary_df: pd.DataFrame,
    play_plan_df: pd.DataFrame,
    current_skip_df: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("CORE 025 SKIP ENGINE v3.1 SUMMARY")
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append("")
    lines.append(f"Transition events: {len(transitions_df):,}")
    lines.append(f"Core 025 hits: {int(transitions_df['next_is_core025_hit'].sum()):,}")
    lines.append(f"Core 025 base rate: {float(transitions_df['next_is_core025_hit'].mean()):.4f}")
    lines.append("")

    lines.append("Top negative traits:")
    for _, r in negative_traits_df.head(12).iterrows():
        lines.append(f"  - {r['trait']} | support={int(r['support'])} | hit_rate={r['hit_rate']:.4f} | gain={r['gain_vs_base']:.4f}")

    lines.append("")
    lines.append("Top skip buckets:")
    for _, r in skip_buckets_df.head(12).iterrows():
        lines.append(f"  - bucket_id={int(r['bucket_id'])} | support={int(r['support'])} | hit_rate={r['hit_rate']:.4f} | {r['bucket_traits']}")

    lines.append("")
    lines.append("Historical class summary:")
    for _, r in class_summary_df.iterrows():
        lines.append(f"  - {r['skip_class']} | events={int(r['events'])} | hits={int(r['core025_hits'])} | hit_rate={r['hit_rate']:.4f}")

    lines.append("")
    lines.append("Historical play plans:")
    for _, r in play_plan_df.iterrows():
        lines.append(f"  - {r['plan']} | plays_saved={int(r['plays_saved'])} | hits_kept={int(r['core025_hits_kept'])} | hit_retention={r['hit_retention_pct']:.4f}")

    lines.append("")
    lines.append("Current highest skip scores:")
    for _, r in current_skip_df.head(12).iterrows():
        lines.append(f"  - {r['stream_id']} | seed={r['seed']} | skip_score={r['skip_score']:.3f} | class={r['skip_class']} | fired={r['fired_skip_bucket_ids']}")

    return "\n".join(lines)


def run_pipeline(
    main_raw_df: pd.DataFrame,
    last24_raw_df: Optional[pd.DataFrame],
    min_trait_support: int,
    bucket_min_support: int,
    bucket_top_k: int,
    bucket_max_depth: int,
    max_hit_rate_allowed: float,
    top_skip_buckets_to_apply: int,
    skip_score_threshold: float,
) -> Dict[str, object]:
    main_history = prepare_history(main_raw_df)
    last24_history = prepare_history(last24_raw_df) if last24_raw_df is not None else None

    transitions_df = build_transition_events(main_history)
    feat_df = build_feature_table(transitions_df)

    negative_traits_df = mine_negative_traits(feat_df, min_support=int(min_trait_support))
    skip_buckets_df = build_stacked_skip_buckets(
        df=feat_df,
        negative_traits=negative_traits_df,
        bucket_min_support=int(bucket_min_support),
        bucket_top_k=int(bucket_top_k),
        bucket_max_depth=int(bucket_max_depth),
        max_hit_rate_allowed=float(max_hit_rate_allowed),
    )

    scored_hist_df = build_historical_scoring_table(
        feat_df=feat_df,
        skip_buckets=skip_buckets_df,
        top_skip_buckets_to_apply=int(top_skip_buckets_to_apply),
        skip_score_threshold=float(skip_score_threshold),
    )
    class_summary_df = summarize_scoring(scored_hist_df)
    play_plan_df = summarize_play_plans(scored_hist_df)

    stream_profiles_df = compute_stream_profiles(transitions_df)
    current_seeds_df = get_current_seed_rows(main_history, last24_history)
    current_skip_df = build_current_skip_table(
        current_seeds=current_seeds_df,
        stream_profiles=stream_profiles_df,
        skip_buckets=skip_buckets_df,
        top_skip_buckets_to_apply=int(top_skip_buckets_to_apply),
        skip_score_threshold=float(skip_score_threshold),
    )

    summary_text = build_summary_text(
        transitions_df=transitions_df,
        negative_traits_df=negative_traits_df,
        skip_buckets_df=skip_buckets_df,
        class_summary_df=class_summary_df,
        play_plan_df=play_plan_df,
        current_skip_df=current_skip_df,
    )

    return {
        "main_history": main_history,
        "last24_history": last24_history,
        "transitions": transitions_df,
        "features": feat_df,
        "negative_traits": negative_traits_df,
        "skip_buckets": skip_buckets_df,
        "historical_scoring": scored_hist_df,
        "class_summary": class_summary_df,
        "play_plans": play_plan_df,
        "stream_profiles": stream_profiles_df,
        "current_skip": current_skip_df,
        "summary_text": summary_text,
        "completed_at_utc": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
    }


# ---------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------

def run_streamlit_app() -> None:
    st.set_page_config(page_title="Core025 Skip Engine v3.1", layout="wide")

    if "skip_engine_v31_results" not in st.session_state:
        st.session_state["skip_engine_v31_results"] = None

    st.title("Core025 Skip Engine v3.1 (Control + Scoring)")
    st.caption("Deep negative-trait discovery with sidebar controls, current scoring, and historical play/skip evaluation.")

    with st.sidebar:
        st.header("Mining controls")
        min_trait_support = st.number_input("Minimum trait support", min_value=3, value=12, step=1)
        bucket_min_support = st.number_input("Skip bucket minimum support", min_value=10, value=20, step=5)
        bucket_top_k = st.number_input("Skip bucket top K traits", min_value=5, value=40, step=5)
        bucket_max_depth = st.number_input("Skip bucket max depth", min_value=1, value=4, step=1)
        max_hit_rate_allowed = st.number_input("Max hit rate allowed for skip trait/bucket", min_value=0.0, max_value=0.05, value=0.0015, step=0.0005, format="%.4f")

        st.header("Scoring controls")
        top_skip_buckets_to_apply = st.number_input("Top skip buckets to apply", min_value=1, value=10, step=1)
        skip_score_threshold = st.slider("Skip score threshold", min_value=0.0, max_value=1.0, value=0.60, step=0.01)
        rows_to_show = st.number_input("Rows to display per table", min_value=5, value=25, step=5)

        if st.button("Clear stored results"):
            st.session_state["skip_engine_v31_results"] = None
            st.rerun()

    st.subheader("Upload files")
    main_file = st.file_uploader(
        "Required main history file (full history)",
        type=["txt", "tsv", "csv", "xlsx", "xls"],
        key="skip_v31_main_uploader",
    )
    last24_file = st.file_uploader(
        "Optional last 24 file (same raw-history format)",
        type=["txt", "tsv", "csv", "xlsx", "xls"],
        key="skip_v31_last24_uploader",
    )

    if main_file is None:
        st.info("Upload the main history file to begin.")
        return

    try:
        main_raw_df = read_uploaded_table(main_file)
        last24_raw_df = read_uploaded_table(last24_file) if last24_file is not None else None
    except Exception as e:
        st.error(f"Could not read uploaded file(s): {e}")
        return

    st.subheader("Raw file preview")
    st.write(f"Main file: {main_file.name}")
    st.write(f"Rows: {len(main_raw_df):,} | Columns: {len(main_raw_df.columns)}")
    st.dataframe(safe_display_df(main_raw_df, 10), use_container_width=True)
    if last24_raw_df is not None:
        st.write(f"Optional last 24 file: {last24_file.name}")
        st.write(f"Rows: {len(last24_raw_df):,} | Columns: {len(last24_raw_df.columns)}")
        st.dataframe(safe_display_df(last24_raw_df, 10), use_container_width=True)

    if st.button("Run Core025 Skip Engine v3.1", type="primary"):
        try:
            with st.spinner("Mining negative traits, building buckets, and scoring skip decisions..."):
                results = run_pipeline(
                    main_raw_df=main_raw_df,
                    last24_raw_df=last24_raw_df,
                    min_trait_support=int(min_trait_support),
                    bucket_min_support=int(bucket_min_support),
                    bucket_top_k=int(bucket_top_k),
                    bucket_max_depth=int(bucket_max_depth),
                    max_hit_rate_allowed=float(max_hit_rate_allowed),
                    top_skip_buckets_to_apply=int(top_skip_buckets_to_apply),
                    skip_score_threshold=float(skip_score_threshold),
                )
            st.session_state["skip_engine_v31_results"] = results
            st.rerun()
        except Exception as e:
            st.exception(e)
            return

    results = st.session_state.get("skip_engine_v31_results")
    if results is None:
        st.info("Click the run button after uploading the main history file.")
        return

    st.success(f"Completed at UTC: {results['completed_at_utc']}")

    transitions_df = results["transitions"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transition events", f"{len(transitions_df):,}")
    c2.metric("Core025 hits", f"{int(transitions_df['next_is_core025_hit'].sum()):,}")
    c3.metric("Base rate", f"{float(transitions_df['next_is_core025_hit'].mean()):.4f}")
    c4.metric("Buckets found", f"{len(results['skip_buckets']):,}")

    st.subheader("Summary")
    st.text_area("Summary text", results["summary_text"], height=420)
    st.download_button(
        "Download summary TXT",
        data=results["summary_text"].encode("utf-8"),
        file_name="core025_skip_engine_v3_1_summary__2026-03-25.txt",
        mime="text/plain",
    )

    st.markdown("## Historical scoring")
    tab1, tab2, tab3 = st.tabs(["Class summary", "Play plans", "Event scoring"])

    with tab1:
        st.dataframe(safe_display_df(results["class_summary"], int(rows_to_show)), use_container_width=True)
        st.download_button(
            "Download class summary CSV",
            data=df_to_csv_bytes(results["class_summary"]),
            file_name="core025_skip_engine_v3_1_class_summary__2026-03-25.csv",
            mime="text/csv",
        )

    with tab2:
        st.dataframe(safe_display_df(results["play_plans"], int(rows_to_show)), use_container_width=True)
        st.download_button(
            "Download play plans CSV",
            data=df_to_csv_bytes(results["play_plans"]),
            file_name="core025_skip_engine_v3_1_play_plans__2026-03-25.csv",
            mime="text/csv",
        )

    with tab3:
        st.dataframe(safe_display_df(results["historical_scoring"], int(rows_to_show)), use_container_width=True)
        st.download_button(
            "Download historical scoring CSV",
            data=df_to_csv_bytes(results["historical_scoring"]),
            file_name="core025_skip_engine_v3_1_historical_scoring__2026-03-25.csv",
            mime="text/csv",
        )

    st.markdown("## Current skip scoring")
    st.dataframe(safe_display_df(results["current_skip"], int(rows_to_show)), use_container_width=True)
    st.download_button(
        "Download current skip scoring CSV",
        data=df_to_csv_bytes(results["current_skip"]),
        file_name="core025_skip_engine_v3_1_current_skip__2026-03-25.csv",
        mime="text/csv",
    )

    st.markdown("## Top negative traits")
    st.dataframe(safe_display_df(results["negative_traits"], int(rows_to_show)), use_container_width=True)
    st.download_button(
        "Download negative traits CSV",
        data=df_to_csv_bytes(results["negative_traits"]),
        file_name="core025_skip_engine_v3_1_negative_traits__2026-03-25.csv",
        mime="text/csv",
    )

    st.markdown("## Skip buckets")
    st.dataframe(safe_display_df(results["skip_buckets"], int(rows_to_show)), use_container_width=True)
    st.download_button(
        "Download skip buckets CSV",
        data=df_to_csv_bytes(results["skip_buckets"]),
        file_name="core025_skip_engine_v3_1_skip_buckets__2026-03-25.csv",
        mime="text/csv",
    )


def main():
    if has_streamlit_context():
        run_streamlit_app()
    else:
        raise SystemExit("Run this file with: streamlit run core025_skip_engine_app_v3_1__2026-03-25.py")


if __name__ == "__main__":
    main()
