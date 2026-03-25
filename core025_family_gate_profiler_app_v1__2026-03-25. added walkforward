#!/usr/bin/env python3
"""
core025_family_gate_profiler_app_v2__2026-03-25.py

Stage 1 app for Core 025 family gating.

What is new in v2:
- Adds a FAST retrospective backtest panel
- Shows how SKIP / NEUTRAL / MORE LIVE performed on historical events
- Shows hit retention vs plays saved for simple gate plans
- Keeps optional last-24 uploader for current stream scoring

Important note:
- The new backtest panel is FAST and retrospective.
- It is useful for quickly validating whether the gate has signal.
- It is not yet a strict walk-forward/no-lookahead backtest.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None


CORE025_SET = {"0025", "0225", "0255"}
MIRROR_PAIRS = {(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)}
DIGITS = list(range(10))


# ---------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------

def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: Dict[str, int] = {}
    new_cols: List[str] = []
    for col in df.columns:
        name = str(col)
        if name not in seen:
            seen[name] = 0
            new_cols.append(name)
        else:
            seen[name] += 1
            new_cols.append(f"{name}__dup{seen[name]}")
    out = df.copy()
    out.columns = new_cols
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


def has_streamlit_context() -> bool:
    if st is None:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def percentile_rank_series(s: pd.Series) -> pd.Series:
    if len(s) == 0:
        return s
    return s.rank(method="average", pct=True)


# ---------------------------------------------------------
# Raw history parsing
# ---------------------------------------------------------

def normalize_result_to_4digits(result_text: str) -> Optional[str]:
    if pd.isna(result_text):
        return None
    digits = re.findall(r"\d", str(result_text))
    if len(digits) < 4:
        return None
    return "".join(digits[:4])


def canonical_member_family(result4: str) -> Optional[str]:
    if result4 is None:
        return None
    sorted4 = "".join(sorted(result4))
    if sorted4 in CORE025_SET:
        return sorted4
    return None


def parse_date_col(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


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


def prepare_raw_history(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = dedupe_columns(df_raw.copy())

    if list(df.columns) == [0, 1, 2, 3] or len(df.columns) == 4:
        c0, c1, c2, c3 = list(df.columns)[:4]
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

    df["date_dt"] = parse_date_col(df["date"])
    df["result4"] = df["result_raw"].apply(normalize_result_to_4digits)
    df["stream_id"] = df["jurisdiction"].astype(str).str.strip() + " | " + df["game"].astype(str).str.strip()
    df["core025_member"] = df["result4"].apply(canonical_member_family)
    df["is_core025_hit"] = df["core025_member"].notna().astype(int)

    df = df.dropna(subset=["result4"]).copy()
    df = df.reset_index(drop=True)
    df["file_order"] = np.arange(len(df))
    df["chron_order"] = -df["file_order"]
    return dedupe_columns(df)


# ---------------------------------------------------------
# Build per-stream transitions
# ---------------------------------------------------------

def build_transition_events(history_df: pd.DataFrame) -> pd.DataFrame:
    sort_df = history_df.sort_values(["stream_id", "date_dt", "chron_order"], ascending=[True, True, True]).copy()

    rows: List[Dict[str, object]] = []

    for stream_id, g in sort_df.groupby("stream_id", sort=False):
        g = g.reset_index(drop=True).copy()
        if len(g) < 2:
            continue

        hit_positions = [i for i, v in enumerate(g["is_core025_hit"].tolist()) if int(v) == 1]

        for i in range(1, len(g)):
            prev_row = g.iloc[i - 1]
            cur_row = g.iloc[i]

            last_hit_before_prev = None
            for hp in hit_positions:
                if hp < i:
                    last_hit_before_prev = hp
                else:
                    break
            current_gap_before_event = i - 1 - last_hit_before_prev if last_hit_before_prev is not None else i

            last_50_start = max(0, i - 50)
            recent_50_rate = float(g.iloc[last_50_start:i]["is_core025_hit"].mean()) if i > last_50_start else 0.0

            rows.append({
                "stream_id": stream_id,
                "jurisdiction": cur_row["jurisdiction"],
                "game": cur_row["game"],
                "event_date": cur_row["date_dt"],
                "seed": prev_row["result4"],
                "next_result4": cur_row["result4"],
                "next_core025_member": cur_row["core025_member"] if pd.notna(cur_row["core025_member"]) else "",
                "next_is_core025_hit": int(cur_row["is_core025_hit"]),
                "seed_date": prev_row["date_dt"],
                "seed_is_core025_hit": int(prev_row["is_core025_hit"]),
                "current_gap_before_event": int(current_gap_before_event),
                "recent_50_hit_rate_before_event": recent_50_rate,
                "stream_event_index": i,
            })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError("No usable transitions were created. Check the uploaded history.")
    return dedupe_columns(out)


# ---------------------------------------------------------
# Seed features
# ---------------------------------------------------------

def digit_list(seed: str) -> List[int]:
    return [int(ch) for ch in str(seed)]


def as_pair_tokens(seed: str) -> List[str]:
    ds = list(seed)
    out = []
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            out.append("".join(sorted((ds[i], ds[j]))))
    return out


def as_ordered_adj_pairs(seed: str) -> List[str]:
    return [seed[i:i + 2] for i in range(len(seed) - 1)]


def as_unordered_adj_pairs(seed: str) -> List[str]:
    return ["".join(sorted(seed[i:i + 2])) for i in range(len(seed) - 1)]


def compute_seed_features(seed: str) -> Dict[str, object]:
    d = digit_list(seed)
    cnt = Counter(d)
    s = sum(d)
    spread = max(d) - min(d)
    parity = "".join("E" if x % 2 == 0 else "O" for x in d)
    highlow = "".join("H" if x >= 5 else "L" for x in d)

    consec_links = 0
    unique_sorted = sorted(set(d))
    for a, b in zip(unique_sorted[:-1], unique_sorted[1:]):
        if b - a == 1:
            consec_links += 1

    mirrorpair_cnt = sum(1 for a, b in MIRROR_PAIRS if a in cnt and b in cnt)

    pairwise_absdiff = []
    for i in range(4):
        for j in range(i + 1, 4):
            pairwise_absdiff.append(abs(d[i] - d[j]))
    adj_absdiff = [abs(d[i] - d[i + 1]) for i in range(3)]

    features: Dict[str, object] = {
        "feat_seed": seed,
        "seed_sum": s,
        "seed_sum_lastdigit": s % 10,
        "seed_sum_mod3": s % 3,
        "seed_sum_mod4": s % 4,
        "seed_sum_mod5": s % 5,
        "seed_spread": spread,
        "seed_unique_digits": len(cnt),
        "seed_has_pair": int(max(cnt.values()) >= 2),
        "seed_no_pair": int(max(cnt.values()) == 1),
        "seed_has_trip": int(max(cnt.values()) >= 3),
        "seed_has_quad": int(max(cnt.values()) >= 4),
        "seed_even_cnt": int(sum(x % 2 == 0 for x in d)),
        "seed_odd_cnt": int(sum(x % 2 == 1 for x in d)),
        "seed_high_cnt": int(sum(x >= 5 for x in d)),
        "seed_low_cnt": int(sum(x <= 4 for x in d)),
        "seed_consec_links": consec_links,
        "seed_mirrorpair_cnt": mirrorpair_cnt,
        "seed_pairwise_absdiff_sum": int(sum(pairwise_absdiff)),
        "seed_pairwise_absdiff_max": int(max(pairwise_absdiff)),
        "seed_pairwise_absdiff_min": int(min(pairwise_absdiff)),
        "seed_adj_absdiff_sum": int(sum(adj_absdiff)),
        "seed_adj_absdiff_max": int(max(adj_absdiff)),
        "seed_adj_absdiff_min": int(min(adj_absdiff)),
        "seed_pos1": d[0],
        "seed_pos2": d[1],
        "seed_pos3": d[2],
        "seed_pos4": d[3],
        "seed_first_last_sum": d[0] + d[3],
        "seed_middle_sum": d[1] + d[2],
        "seed_absdiff_outer_inner": abs((d[0] + d[3]) - (d[1] + d[2])),
        "seed_parity_pattern": parity,
        "seed_highlow_pattern": highlow,
        "seed_sorted": "".join(map(str, sorted(d))),
        "seed_pair_tokens": "|".join(sorted(as_pair_tokens(seed))),
        "seed_adj_pairs_ordered": "|".join(as_ordered_adj_pairs(seed)),
        "seed_adj_pairs_unordered": "|".join(sorted(as_unordered_adj_pairs(seed))),
        "seed_outer_equal": int(d[0] == d[3]),
        "seed_inner_equal": int(d[1] == d[2]),
        "seed_sum_even": int(s % 2 == 0),
    }

    for k in DIGITS:
        features[f"seed_has{k}"] = int(k in cnt)
        features[f"seed_cnt{k}"] = int(cnt.get(k, 0))

    shape = "".join(map(str, sorted(cnt.values(), reverse=True)))
    features["seed_repeat_shape"] = {
        "1111": "all_unique",
        "211": "one_pair",
        "22": "two_pair",
        "31": "trip",
        "4": "quad",
    }.get(shape, f"shape_{shape}")

    features["cnt_0_3"] = int(sum(0 <= x <= 3 for x in d))
    features["cnt_4_6"] = int(sum(4 <= x <= 6 for x in d))
    features["cnt_7_9"] = int(sum(7 <= x <= 9 for x in d))

    pair_counts = Counter(as_pair_tokens(seed))
    for a in range(10):
        for b in range(a, 10):
            tok = f"{a}{b}"
            features[f"pair_has_{tok}"] = int(pair_counts.get(tok, 0) > 0)

    ordered_adj = as_ordered_adj_pairs(seed)
    for a in range(10):
        for b in range(10):
            tok = f"{a}{b}"
            features[f"adj_ord_has_{tok}"] = int(tok in ordered_adj)

    return features


# ---------------------------------------------------------
# Trait matrix / scoring for binary family-hit task
# ---------------------------------------------------------

def bin_numeric_series(s: pd.Series, prefix: str) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    vals = pd.to_numeric(s, errors="coerce")
    if vals.notna().sum() == 0:
        return out

    uniques = sorted(set(vals.dropna().astype(float).tolist()))
    if len(uniques) <= 20:
        for u in uniques:
            label = str(int(u)) if float(u).is_integer() else str(u)
            out[f"{prefix}=={label}"] = vals == u

    qs = [0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9]
    quantiles = sorted(set(float(vals.quantile(q)) for q in qs if vals.notna().sum() >= 10))
    for q in quantiles:
        if np.isnan(q):
            continue
        label = int(q) if float(q).is_integer() else round(float(q), 3)
        out[f"{prefix}<={label}"] = vals <= q
        out[f"{prefix}>={label}"] = vals >= q

    if len(uniques) <= 20 and all(float(x).is_integer() for x in uniques):
        int_uniques = [int(x) for x in uniques]
        for lo in int_uniques:
            for hi in int_uniques:
                if lo < hi and (hi - lo) <= 3:
                    out[f"{prefix}_in[{lo},{hi}]"] = (vals >= lo) & (vals <= hi)
    return out


def categorical_series_traits(s: pd.Series, prefix: str) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    vc = s.astype(str).value_counts(dropna=False)
    for v, n in vc.items():
        if n >= 5:
            out[f"{prefix}=={v}"] = s.astype(str) == str(v)
    return out


def build_trait_matrix(df_feat: pd.DataFrame, mine_level: str = "standard") -> pd.DataFrame:
    trait_cols: Dict[str, pd.Series] = {}

    numeric_cols = [
        c for c in df_feat.columns
        if (c.startswith("seed_") or c.startswith("cnt_") or c in ["current_gap_before_event", "recent_50_hit_rate_before_event"])
        and c not in {"feat_seed"}
    ]
    categorical_cols = [
        "seed_parity_pattern",
        "seed_highlow_pattern",
        "seed_repeat_shape",
        "seed_sorted",
    ]

    for c in numeric_cols:
        ser = df_feat[c]
        if pd.api.types.is_numeric_dtype(ser):
            trait_cols.update(bin_numeric_series(ser, c))

    for c in categorical_cols:
        if c in df_feat.columns:
            trait_cols.update(categorical_series_traits(df_feat[c], c))

    sparse_prefixes = ("pair_has_", "adj_ord_has_", "seed_has", "seed_cnt")
    for c in df_feat.columns:
        if c.startswith(sparse_prefixes):
            ser = pd.to_numeric(df_feat[c], errors="coerce").fillna(0).astype(int)
            if ser.sum() >= 8:
                trait_cols[c] = ser.astype(bool)

    if mine_level == "expanded":
        for left, right in [
            ("seed_has_pair", "seed_sum_even"),
            ("seed_has9", "seed_repeat_shape==one_pair"),
            ("seed_has2", "seed_has5"),
            ("seed_sum_mod4_in[1,2]", "cnt_0_3_in[3,4]"),
        ]:
            if left in trait_cols and right in trait_cols:
                trait_cols[f"{left} AND {right}"] = trait_cols[left] & trait_cols[right]

    trait_df = pd.DataFrame(trait_cols, index=df_feat.index).astype(bool)
    return dedupe_columns(trait_df)


def score_traits_binary(trait_df: pd.DataFrame, y_hit: pd.Series) -> pd.DataFrame:
    target = y_hit.astype(int)
    rows: List[Dict[str, object]] = []
    total_hits = int(target.sum())
    total_n = int(len(target))
    base_rate = total_hits / total_n if total_n else 0.0

    for trait in trait_df.columns:
        mask = trait_df[trait].fillna(False).astype(bool)
        support = int(mask.sum())
        if support == 0 or support == total_n:
            continue

        hits_true = int(target[mask].sum())
        misses_true = int(support - hits_true)
        hit_rate_true = hits_true / support if support else 0.0

        inv = ~mask
        support_false = int(inv.sum())
        hits_false = int(target[inv].sum())
        misses_false = int(support_false - hits_false)
        hit_rate_false = hits_false / support_false if support_false else 0.0

        rows.append({
            "trait": trait,
            "support": support,
            "support_pct": support / total_n if total_n else 0.0,
            "hits_true": hits_true,
            "misses_true": misses_true,
            "hit_rate_true": hit_rate_true,
            "support_false": support_false,
            "hits_false": hits_false,
            "misses_false": misses_false,
            "hit_rate_false": hit_rate_false,
            "base_rate_core025": base_rate,
            "lift_vs_base": hit_rate_true - base_rate,
            "precision_gap": hit_rate_true - hit_rate_false,
            "is_negative_separator": int(hits_true == 0 and support > 0),
            "is_positive_trait": int(hit_rate_true > base_rate),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["precision_gap", "hit_rate_true", "support"], ascending=[False, False, False]).reset_index(drop=True)
    return out


# ---------------------------------------------------------
# Stream profiles / live gate scoring
# ---------------------------------------------------------

def compute_stream_profiles(transitions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for stream_id, g in transitions.groupby("stream_id", sort=False):
        g = g.sort_values(["event_date", "stream_event_index"]).reset_index(drop=True)
        hits = int(g["next_is_core025_hit"].sum())
        n = int(len(g))
        base_rate = hits / n if n else 0.0

        hit_positions = [i for i, v in enumerate(g["next_is_core025_hit"].tolist()) if int(v) == 1]
        gaps = []
        if len(hit_positions) >= 2:
            for a, b in zip(hit_positions[:-1], hit_positions[1:]):
                gaps.append(b - a)

        current_gap = int(g.iloc[-1]["stream_event_index"] - hit_positions[-1]) if len(hit_positions) > 0 else int(g.iloc[-1]["stream_event_index"])
        due_pct = 0.0
        if len(gaps) > 0:
            due_pct = float(sum(current_gap >= gp for gp in gaps) / len(gaps))

        recent_50_rate = float(g.tail(50)["next_is_core025_hit"].mean()) if len(g) > 0 else 0.0

        rows.append({
            "stream_id": stream_id,
            "jurisdiction": g.iloc[-1]["jurisdiction"],
            "game": g.iloc[-1]["game"],
            "events": n,
            "core025_hits": hits,
            "core025_hit_rate": base_rate,
            "avg_gap_between_hits": float(np.mean(gaps)) if len(gaps) > 0 else np.nan,
            "median_gap_between_hits": float(np.median(gaps)) if len(gaps) > 0 else np.nan,
            "max_gap_between_hits": float(np.max(gaps)) if len(gaps) > 0 else np.nan,
            "current_gap_since_last_hit": current_gap,
            "due_percentile_vs_stream_history": due_pct,
            "recent_50_core025_rate": recent_50_rate,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["core025_hit_rate", "events"], ascending=[False, False]).reset_index(drop=True)
    return dedupe_columns(out)


def get_current_seed_rows(main_history: pd.DataFrame, last24_history: Optional[pd.DataFrame]) -> pd.DataFrame:
    source = last24_history if last24_history is not None and len(last24_history) > 0 else main_history
    source = source.sort_values(["stream_id", "date_dt", "chron_order"], ascending=[True, True, True]).copy()
    latest = source.groupby("stream_id", sort=False).tail(1).copy()
    latest = latest[["stream_id", "jurisdiction", "game", "date_dt", "result4"]].rename(columns={
        "date_dt": "seed_date",
        "result4": "seed",
    })
    return dedupe_columns(latest.reset_index(drop=True))


def build_live_gate_table(
    current_seeds: pd.DataFrame,
    stream_profiles: pd.DataFrame,
    positive_traits: List[str],
    negative_traits: List[str],
    mine_level: str = "standard",
    weight_stream_rate: float = 0.35,
    weight_due: float = 0.20,
    weight_recent50: float = 0.15,
    weight_positive_trait_fire: float = 0.20,
    weight_negative_trait_fire: float = 0.10,
    skip_threshold: float = 0.35,
    live_threshold: float = 0.60,
) -> pd.DataFrame:
    merged = current_seeds.merge(stream_profiles, on=["stream_id", "jurisdiction", "game"], how="left")
    feat_rows = [compute_seed_features(seed) for seed in merged["seed"].astype(str)]
    df_feat = pd.DataFrame(feat_rows)
    df_feat = dedupe_columns(pd.concat([merged.reset_index(drop=True), df_feat.reset_index(drop=True)], axis=1))
    trait_df = build_trait_matrix(df_feat, mine_level=mine_level)

    pos_fires = []
    neg_fires = []
    for i in trait_df.index:
        pos_count = 0
        neg_count = 0
        for t in positive_traits:
            if t in trait_df.columns and bool(trait_df.loc[i, t]):
                pos_count += 1
        for t in negative_traits:
            if t in trait_df.columns and bool(trait_df.loc[i, t]):
                neg_count += 1
        pos_fires.append(pos_count)
        neg_fires.append(neg_count)

    df_feat["positive_trait_fire_count"] = pos_fires
    df_feat["negative_trait_fire_count"] = neg_fires

    df_feat["stream_rate_pct"] = percentile_rank_series(df_feat["core025_hit_rate"].fillna(0))
    df_feat["recent50_pct"] = percentile_rank_series(df_feat["recent_50_core025_rate"].fillna(0))
    df_feat["due_pct_clean"] = df_feat["due_percentile_vs_stream_history"].fillna(0)
    df_feat["positive_trait_pct"] = percentile_rank_series(df_feat["positive_trait_fire_count"].fillna(0))
    df_feat["negative_trait_pct"] = percentile_rank_series(df_feat["negative_trait_fire_count"].fillna(0))

    df_feat["gate_score"] = (
        weight_stream_rate * df_feat["stream_rate_pct"].fillna(0) +
        weight_due * df_feat["due_pct_clean"].fillna(0) +
        weight_recent50 * df_feat["recent50_pct"].fillna(0) +
        weight_positive_trait_fire * df_feat["positive_trait_pct"].fillna(0) -
        weight_negative_trait_fire * df_feat["negative_trait_pct"].fillna(0)
    ).clip(lower=0, upper=1)

    def classify(score: float) -> str:
        if score < skip_threshold:
            return "LIKELY SKIP"
        if score < live_threshold:
            return "NEUTRAL"
        return "MORE LIVE"

    df_feat["gate_class"] = df_feat["gate_score"].apply(classify)

    keep_cols = [
        "stream_id", "jurisdiction", "game", "seed_date", "seed",
        "events", "core025_hits", "core025_hit_rate",
        "current_gap_since_last_hit", "due_percentile_vs_stream_history",
        "recent_50_core025_rate",
        "positive_trait_fire_count", "negative_trait_fire_count",
        "gate_score", "gate_class",
    ]
    out = df_feat[keep_cols].copy()
    out = out.sort_values(["gate_score", "core025_hit_rate", "events"], ascending=[False, False, False]).reset_index(drop=True)
    return dedupe_columns(out)


# ---------------------------------------------------------
# FAST retrospective backtest
# ---------------------------------------------------------

def build_retrospective_backtest(
    model_df: pd.DataFrame,
    trait_df: pd.DataFrame,
    stream_profiles: pd.DataFrame,
    positive_traits: List[str],
    negative_traits: List[str],
    weight_stream_rate: float,
    weight_due: float,
    weight_recent50: float,
    weight_positive_trait_fire: float,
    weight_negative_trait_fire: float,
    skip_threshold: float,
    live_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bt = model_df.copy()
    bt = bt.merge(
        stream_profiles[["stream_id", "core025_hit_rate"]],
        on="stream_id",
        how="left",
        suffixes=("", "_stream"),
    )

    pos_fires = []
    neg_fires = []
    for i in trait_df.index:
        pos_count = 0
        neg_count = 0
        for t in positive_traits:
            if t in trait_df.columns and bool(trait_df.loc[i, t]):
                pos_count += 1
        for t in negative_traits:
            if t in trait_df.columns and bool(trait_df.loc[i, t]):
                neg_count += 1
        pos_fires.append(pos_count)
        neg_fires.append(neg_count)

    bt["positive_trait_fire_count"] = pos_fires
    bt["negative_trait_fire_count"] = neg_fires

    # Percentile features for retrospective validation
    bt["stream_rate_pct"] = percentile_rank_series(bt["core025_hit_rate"].fillna(0))
    bt["recent50_pct"] = percentile_rank_series(bt["recent_50_hit_rate_before_event"].fillna(0))
    bt["positive_trait_pct"] = percentile_rank_series(bt["positive_trait_fire_count"].fillna(0))
    bt["negative_trait_pct"] = percentile_rank_series(bt["negative_trait_fire_count"].fillna(0))

    # Gap percentile within stream, using observed event gaps for a fast retrospective test
    bt["gap_pct_within_stream"] = bt.groupby("stream_id")["current_gap_before_event"].transform(lambda s: percentile_rank_series(s.fillna(0)))

    bt["gate_score"] = (
        float(weight_stream_rate) * bt["stream_rate_pct"].fillna(0) +
        float(weight_due) * bt["gap_pct_within_stream"].fillna(0) +
        float(weight_recent50) * bt["recent50_pct"].fillna(0) +
        float(weight_positive_trait_fire) * bt["positive_trait_pct"].fillna(0) -
        float(weight_negative_trait_fire) * bt["negative_trait_pct"].fillna(0)
    ).clip(lower=0, upper=1)

    def classify(score: float) -> str:
        if score < float(skip_threshold):
            return "LIKELY SKIP"
        if score < float(live_threshold):
            return "NEUTRAL"
        return "MORE LIVE"

    bt["gate_class"] = bt["gate_score"].apply(classify)

    class_rows = []
    total_events = int(len(bt))
    total_hits = int(bt["next_is_core025_hit"].sum())

    for cls in ["LIKELY SKIP", "NEUTRAL", "MORE LIVE"]:
        sub = bt[bt["gate_class"] == cls].copy()
        events = int(len(sub))
        hits = int(sub["next_is_core025_hit"].sum()) if len(sub) else 0
        class_rows.append({
            "gate_class": cls,
            "events": events,
            "events_pct": events / total_events if total_events else 0.0,
            "core025_hits": hits,
            "hit_rate": hits / events if events else 0.0,
            "hit_share_of_all_hits": hits / total_hits if total_hits else 0.0,
            "avg_gate_score": float(sub["gate_score"].mean()) if len(sub) else np.nan,
        })

    class_summary = pd.DataFrame(class_rows)

    # Simple play plans
    plan_rows = []
    plans = {
        "Play MORE LIVE only": bt["gate_class"] == "MORE LIVE",
        "Play NEUTRAL + MORE LIVE": bt["gate_class"].isin(["NEUTRAL", "MORE LIVE"]),
        "Skip LIKELY SKIP only": bt["gate_class"] != "LIKELY SKIP",
        "Play everything": pd.Series([True] * len(bt), index=bt.index),
    }

    for plan_name, play_mask in plans.items():
        played = bt[play_mask].copy()
        skipped = bt[~play_mask].copy()
        plays = int(len(played))
        hits_kept = int(played["next_is_core025_hit"].sum()) if len(played) else 0
        hits_skipped = int(skipped["next_is_core025_hit"].sum()) if len(skipped) else 0
        plan_rows.append({
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

    plan_summary = pd.DataFrame(plan_rows)
    return dedupe_columns(bt), dedupe_columns(class_summary), dedupe_columns(plan_summary)


# ---------------------------------------------------------
# App summary text
# ---------------------------------------------------------

def build_summary_text(
    raw_history: pd.DataFrame,
    transitions: pd.DataFrame,
    stream_profiles: pd.DataFrame,
    trait_scores: pd.DataFrame,
    live_gate: pd.DataFrame,
    backtest_class_summary: pd.DataFrame,
    backtest_plan_summary: pd.DataFrame,
    positive_traits: List[str],
    negative_traits: List[str],
) -> str:
    lines = []
    lines.append("CORE 025 FAMILY GATE PROFILER SUMMARY")
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append("")
    lines.append(f"Raw history rows: {len(raw_history):,}")
    lines.append(f"Transition events: {len(transitions):,}")
    lines.append(f"Distinct streams: {stream_profiles['stream_id'].nunique():,}")
    lines.append(f"Core 025 family hits in transitions: {int(transitions['next_is_core025_hit'].sum()):,}")
    lines.append(f"Core 025 base rate in transitions: {float(transitions['next_is_core025_hit'].mean()):.4f}")
    lines.append("")
    lines.append("Top positive family-hit traits:")
    if len(positive_traits) == 0:
        lines.append("  - none")
    else:
        for t in positive_traits[:15]:
            row = trait_scores.loc[trait_scores["trait"] == t].head(1)
            if len(row) == 1:
                r = row.iloc[0]
                lines.append(f"  - {t} | support={int(r['support'])} | hit_rate_true={r['hit_rate_true']:.4f} | gap={r['precision_gap']:.4f}")

    lines.append("")
    lines.append("Top negative family-hit traits:")
    if len(negative_traits) == 0:
        lines.append("  - none")
    else:
        for t in negative_traits[:15]:
            row = trait_scores.loc[trait_scores["trait"] == t].head(1)
            if len(row) == 1:
                r = row.iloc[0]
                lines.append(f"  - {t} | support={int(r['support'])} | hit_rate_true={r['hit_rate_true']:.4f} | gap={r['precision_gap']:.4f}")

    lines.append("")
    lines.append("Top current streams by gate score:")
    for _, r in live_gate.head(15).iterrows():
        lines.append(
            f"  - {r['stream_id']} | seed={r['seed']} | gate_score={r['gate_score']:.3f} | "
            f"class={r['gate_class']} | stream_rate={r['core025_hit_rate']:.4f} | "
            f"due_pct={r['due_percentile_vs_stream_history']:.3f}"
        )

    lines.append("")
    lines.append("FAST retrospective backtest by gate class:")
    for _, r in backtest_class_summary.iterrows():
        lines.append(
            f"  - {r['gate_class']} | events={int(r['events'])} | hits={int(r['core025_hits'])} | "
            f"hit_rate={r['hit_rate']:.4f} | hit_share={r['hit_share_of_all_hits']:.4f}"
        )

    lines.append("")
    lines.append("FAST retrospective backtest play plans:")
    for _, r in backtest_plan_summary.iterrows():
        lines.append(
            f"  - {r['plan']} | plays={int(r['plays'])} | plays_saved={int(r['plays_saved'])} | "
            f"hits_kept={int(r['core025_hits_kept'])} | hits_skipped={int(r['core025_hits_skipped'])} | "
            f"hit_retention={r['hit_retention_pct']:.4f} | hit_rate_on_played={r['hit_rate_on_played_events']:.4f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------
# Main app pipeline
# ---------------------------------------------------------

def run_family_gate_pipeline(
    main_raw_df: pd.DataFrame,
    last24_raw_df: Optional[pd.DataFrame],
    mine_level: str,
    top_positive_n: int,
    top_negative_n: int,
    min_trait_support: int,
    weight_stream_rate: float,
    weight_due: float,
    weight_recent50: float,
    weight_positive_trait_fire: float,
    weight_negative_trait_fire: float,
    skip_threshold: float,
    live_threshold: float,
) -> Dict[str, object]:
    main_history = prepare_raw_history(main_raw_df)
    last24_history = prepare_raw_history(last24_raw_df) if last24_raw_df is not None else None

    transitions = build_transition_events(main_history)
    seed_feat_rows = [compute_seed_features(seed) for seed in transitions["seed"].astype(str)]
    seed_feat = pd.DataFrame(seed_feat_rows)
    model_df = dedupe_columns(pd.concat([transitions.reset_index(drop=True), seed_feat.reset_index(drop=True)], axis=1))
    trait_df = build_trait_matrix(model_df, mine_level=mine_level)
    trait_scores = score_traits_binary(trait_df, model_df["next_is_core025_hit"].astype(int))

    if len(trait_scores) > 0:
        pos_df = trait_scores[
            (trait_scores["support"] >= int(min_trait_support)) &
            (trait_scores["is_positive_trait"] == 1)
        ].sort_values(["precision_gap", "hit_rate_true", "support"], ascending=[False, False, False])
        neg_df = trait_scores[
            (trait_scores["support"] >= int(min_trait_support))
        ].sort_values(["precision_gap", "hit_rate_true", "support"], ascending=[True, True, False])

        positive_traits = pos_df["trait"].head(int(top_positive_n)).tolist()
        negative_traits = neg_df["trait"].head(int(top_negative_n)).tolist()
    else:
        positive_traits = []
        negative_traits = []

    stream_profiles = compute_stream_profiles(transitions)
    current_seeds = get_current_seed_rows(main_history, last24_history)

    live_gate = build_live_gate_table(
        current_seeds=current_seeds,
        stream_profiles=stream_profiles,
        positive_traits=positive_traits,
        negative_traits=negative_traits,
        mine_level=mine_level,
        weight_stream_rate=float(weight_stream_rate),
        weight_due=float(weight_due),
        weight_recent50=float(weight_recent50),
        weight_positive_trait_fire=float(weight_positive_trait_fire),
        weight_negative_trait_fire=float(weight_negative_trait_fire),
        skip_threshold=float(skip_threshold),
        live_threshold=float(live_threshold),
    )

    backtest_events, backtest_class_summary, backtest_plan_summary = build_retrospective_backtest(
        model_df=model_df,
        trait_df=trait_df,
        stream_profiles=stream_profiles,
        positive_traits=positive_traits,
        negative_traits=negative_traits,
        weight_stream_rate=float(weight_stream_rate),
        weight_due=float(weight_due),
        weight_recent50=float(weight_recent50),
        weight_positive_trait_fire=float(weight_positive_trait_fire),
        weight_negative_trait_fire=float(weight_negative_trait_fire),
        skip_threshold=float(skip_threshold),
        live_threshold=float(live_threshold),
    )

    summary_text = build_summary_text(
        raw_history=main_history,
        transitions=transitions,
        stream_profiles=stream_profiles,
        trait_scores=trait_scores,
        live_gate=live_gate,
        backtest_class_summary=backtest_class_summary,
        backtest_plan_summary=backtest_plan_summary,
        positive_traits=positive_traits,
        negative_traits=negative_traits,
    )

    return {
        "main_history": main_history,
        "last24_history": last24_history,
        "transitions": transitions,
        "model_df": model_df,
        "trait_df": trait_df,
        "trait_scores": trait_scores,
        "stream_profiles": stream_profiles,
        "current_seeds": current_seeds,
        "live_gate": live_gate,
        "backtest_events": backtest_events,
        "backtest_class_summary": backtest_class_summary,
        "backtest_plan_summary": backtest_plan_summary,
        "positive_traits": positive_traits,
        "negative_traits": negative_traits,
        "summary_text": summary_text,
        "completed_at_utc": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
    }


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "family_gate_results": None,
        "uploaded_main_name": None,
        "uploaded_last24_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def run_streamlit_app() -> None:
    st.set_page_config(page_title="Core 025 Family Gate Profiler", layout="wide")
    init_session_state()

    st.title("Core 025 Family Gate Profiler")
    st.caption(
        "Stage 1 app: identify streams and seed contexts where a Core 025 family hit "
        "looks weaker, neutral, or more live. This app does not pick the member yet."
    )

    with st.sidebar:
        st.header("Mining settings")
        mine_level = st.selectbox("Mine level", options=["standard", "expanded"], index=1)
        min_trait_support = st.number_input("Minimum trait support", min_value=3, value=12, step=1)
        top_positive_n = st.number_input("Top positive traits to use in gate", min_value=1, value=20, step=1)
        top_negative_n = st.number_input("Top negative traits to use in gate", min_value=1, value=20, step=1)
        top_rows = st.number_input("Rows to display per table", min_value=5, value=25, step=5)

        st.header("Gate score weights")
        weight_stream_rate = st.slider("Weight: stream hit-rate percentile", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
        weight_due = st.slider("Weight: due percentile", min_value=0.0, max_value=1.0, value=0.20, step=0.05)
        weight_recent50 = st.slider("Weight: recent 50 rate percentile", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
        weight_positive_trait_fire = st.slider("Weight: positive trait fires", min_value=0.0, max_value=1.0, value=0.20, step=0.05)
        weight_negative_trait_fire = st.slider("Penalty: negative trait fires", min_value=0.0, max_value=1.0, value=0.10, step=0.05)

        st.header("Gate classes")
        skip_threshold = st.slider("Likely Skip threshold", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
        live_threshold = st.slider("More Live threshold", min_value=0.0, max_value=1.0, value=0.60, step=0.01)

        if st.button("Clear stored results"):
            st.session_state["family_gate_results"] = None
            st.session_state["uploaded_main_name"] = None
            st.session_state["uploaded_last24_name"] = None
            st.rerun()

    st.subheader("Upload files")
    main_file = st.file_uploader(
        "Required main history file (full history)",
        type=["txt", "tsv", "csv", "xlsx", "xls"],
        key="main_history_uploader",
    )
    last24_file = st.file_uploader(
        "Optional last 24 file (same raw-history format)",
        type=["txt", "tsv", "csv", "xlsx", "xls"],
        key="last24_history_uploader",
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

    st.session_state["uploaded_main_name"] = main_file.name
    st.session_state["uploaded_last24_name"] = last24_file.name if last24_file is not None else ""

    st.subheader("Raw file preview")
    st.write(f"Main file: {main_file.name}")
    st.write(f"Rows: {len(main_raw_df):,} | Columns: {len(main_raw_df.columns)}")
    st.dataframe(safe_display_df(main_raw_df, 10), use_container_width=True)

    if last24_raw_df is not None:
        st.write(f"Optional last 24 file: {last24_file.name}")
        st.write(f"Rows: {len(last24_raw_df):,} | Columns: {len(last24_raw_df.columns)}")
        st.dataframe(safe_display_df(last24_raw_df, 10), use_container_width=True)

    run_clicked = st.button("Run Core 025 Family Gate", type="primary")
    if run_clicked:
        try:
            with st.spinner("Building stream profiles, mining family traits, scoring current streams, and running fast backtest..."):
                results = run_family_gate_pipeline(
                    main_raw_df=main_raw_df,
                    last24_raw_df=last24_raw_df,
                    mine_level=mine_level,
                    top_positive_n=int(top_positive_n),
                    top_negative_n=int(top_negative_n),
                    min_trait_support=int(min_trait_support),
                    weight_stream_rate=float(weight_stream_rate),
                    weight_due=float(weight_due),
                    weight_recent50=float(weight_recent50),
                    weight_positive_trait_fire=float(weight_positive_trait_fire),
                    weight_negative_trait_fire=float(weight_negative_trait_fire),
                    skip_threshold=float(skip_threshold),
                    live_threshold=float(live_threshold),
                )
            st.session_state["family_gate_results"] = results
            st.rerun()
        except Exception as e:
            st.exception(e)
            return

    results = st.session_state.get("family_gate_results")
    if results is None:
        st.info("Click 'Run Core 025 Family Gate' after uploading your file(s).")
        return

    st.success(f"Completed at UTC: {results['completed_at_utc']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw history rows", f"{len(results['main_history']):,}")
    c2.metric("Transition events", f"{len(results['transitions']):,}")
    c3.metric("Distinct streams", f"{results['stream_profiles']['stream_id'].nunique():,}")
    c4.metric("Core 025 transition base rate", f"{float(results['transitions']['next_is_core025_hit'].mean()):.4f}")

    st.subheader("Summary")
    st.text_area("Summary text", results["summary_text"], height=420)
    st.download_button(
        "Download summary TXT",
        data=results["summary_text"].encode("utf-8"),
        file_name="core025_family_gate_summary__2026-03-25.txt",
        mime="text/plain",
        key="dl_summary_txt",
    )

    st.markdown("## FAST backtest panel")
    st.caption(
        "This panel is a quick retrospective validation layer. It is useful for checking whether the gate has "
        "real concentration and skip value. It is not yet strict walk-forward."
    )

    tab_bt1, tab_bt2, tab_bt3 = st.tabs(["Backtest class summary", "Backtest play plans", "Backtest event table"])

    with tab_bt1:
        st.dataframe(safe_display_df(results["backtest_class_summary"], int(top_rows)), use_container_width=True)
        st.download_button(
            "Download backtest class summary CSV",
            data=df_to_csv_bytes(results["backtest_class_summary"]),
            file_name="core025_backtest_class_summary__2026-03-25.csv",
            mime="text/csv",
            key="dl_bt_class_summary_csv",
        )

    with tab_bt2:
        st.dataframe(safe_display_df(results["backtest_plan_summary"], int(top_rows)), use_container_width=True)
        st.download_button(
            "Download backtest play plans CSV",
            data=df_to_csv_bytes(results["backtest_plan_summary"]),
            file_name="core025_backtest_play_plans__2026-03-25.csv",
            mime="text/csv",
            key="dl_bt_plan_summary_csv",
        )

    with tab_bt3:
        st.dataframe(safe_display_df(results["backtest_events"], int(top_rows)), use_container_width=True)
        st.download_button(
            "Download backtest events CSV",
            data=df_to_csv_bytes(results["backtest_events"]),
            file_name="core025_backtest_events__2026-03-25.csv",
            mime="text/csv",
            key="dl_bt_events_csv",
        )

    st.markdown("## Current gate scores")
    st.caption("This is the main output for deciding which streams look more skippable, neutral, or more live right now.")
    st.dataframe(safe_display_df(results["live_gate"], int(top_rows)), use_container_width=True)
    st.download_button(
        "Download current gate scores CSV",
        data=df_to_csv_bytes(results["live_gate"]),
        file_name="core025_current_gate_scores__2026-03-25.csv",
        mime="text/csv",
        key="dl_live_gate_csv",
    )

    st.markdown("## Stream profiles")
    st.dataframe(safe_display_df(results["stream_profiles"], int(top_rows)), use_container_width=True)
    st.download_button(
        "Download stream profiles CSV",
        data=df_to_csv_bytes(results["stream_profiles"]),
        file_name="core025_stream_profiles__2026-03-25.csv",
        mime="text/csv",
        key="dl_stream_profiles_csv",
    )

    st.markdown("## Global family-hit trait scores")
    trait_scores = results["trait_scores"]
    pos_df = trait_scores[trait_scores["trait"].isin(results["positive_traits"])].copy()
    neg_df = trait_scores[trait_scores["trait"].isin(results["negative_traits"])].copy()

    tab1, tab2, tab3 = st.tabs(["All traits", "Positive traits used in gate", "Negative traits used in gate"])
    with tab1:
        st.dataframe(safe_display_df(trait_scores, int(top_rows)), use_container_width=True)
        st.download_button(
            "Download all family-hit trait scores CSV",
            data=df_to_csv_bytes(trait_scores),
            file_name="core025_family_hit_trait_scores__2026-03-25.csv",
            mime="text/csv",
            key="dl_trait_scores_csv",
        )
    with tab2:
        st.dataframe(safe_display_df(pos_df, int(top_rows)), use_container_width=True)
        st.download_button(
            "Download positive gate traits CSV",
            data=df_to_csv_bytes(pos_df),
            file_name="core025_positive_gate_traits__2026-03-25.csv",
            mime="text/csv",
            key="dl_positive_traits_csv",
        )
    with tab3:
        st.dataframe(safe_display_df(neg_df, int(top_rows)), use_container_width=True)
        st.download_button(
            "Download negative gate traits CSV",
            data=df_to_csv_bytes(neg_df),
            file_name="core025_negative_gate_traits__2026-03-25.csv",
            mime="text/csv",
            key="dl_negative_traits_csv",
        )

    st.markdown("## Transition event table")
    st.caption("Each row is a seed -> next-result event within a stream. This is the core modeling table for the family gate.")
    st.dataframe(safe_display_df(results["transitions"], int(top_rows)), use_container_width=True)
    st.download_button(
        "Download transition events CSV",
        data=df_to_csv_bytes(results["transitions"]),
        file_name="core025_transition_events__2026-03-25.csv",
        mime="text/csv",
        key="dl_transitions_csv",
    )


def main():
    if has_streamlit_context():
        run_streamlit_app()
    else:
        raise SystemExit("Run this file with: streamlit run core025_family_gate_profiler_app_v2__2026-03-25.py")


if __name__ == "__main__":
    main()
