#!/usr/bin/env python3
"""
core025_skip_engine_app_v2__2026-03-25.py

Purpose
-------
A skip-only engine for Core 025 family play reduction.

This app is intentionally narrower than the family gate profiler.
It does NOT try to predict when Core 025 is likely.
It tries to identify when Core 025 is historically unlikely enough
to justify skipping or reducing plays.

What is fixed in v2
-------------------
- Adds missing percentile_rank_series helper
- Full no-placeholder file
- Same functionality as v1, corrected
"""

from __future__ import annotations

import io
import re
from collections import Counter, defaultdict, deque
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


def safe_result_df(results: Dict[str, object], key: str, fallback_columns: Optional[List[str]] = None) -> pd.DataFrame:
    val = results.get(key, None)
    if isinstance(val, pd.DataFrame):
        return dedupe_columns(val)
    if fallback_columns is None:
        return pd.DataFrame()
    return pd.DataFrame(columns=fallback_columns)


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
            date_col: "date", juris_col: "jurisdiction", game_col: "game", result_col: "result_raw"
        })

    df["date_dt"] = parse_date_col(df["date"])
    df["result4"] = df["result_raw"].apply(normalize_result_to_4digits)
    df["stream_id"] = df["jurisdiction"].astype(str).str.strip() + " | " + df["game"].astype(str).str.strip()
    df["core025_member"] = df["result4"].apply(canonical_member_family)
    df["is_core025_hit"] = df["core025_member"].notna().astype(int)

    df = df.dropna(subset=["result4"]).copy().reset_index(drop=True)
    df["file_order"] = np.arange(len(df))
    df["chron_order"] = -df["file_order"]
    return dedupe_columns(df)


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
        "1111": "all_unique", "211": "one_pair", "22": "two_pair", "31": "trip", "4": "quad"
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


def build_trait_matrix(df_feat: pd.DataFrame, mine_level: str = "expanded") -> pd.DataFrame:
    trait_cols: Dict[str, pd.Series] = {}
    numeric_cols = [c for c in df_feat.columns if (c.startswith("seed_") or c.startswith("cnt_") or c in ["current_gap_before_event", "recent_50_hit_rate_before_event"]) and c not in {"feat_seed"}]
    categorical_cols = ["seed_parity_pattern", "seed_highlow_pattern", "seed_repeat_shape", "seed_sorted"]

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
            ("seed_has2", "seed_has5"),
            ("seed_sum_mod4_in[1,2]", "cnt_0_3_in[3,4]"),
            ("seed_repeat_shape==all_unique", "seed_adj_absdiff_min==1"),
        ]:
            if left in trait_cols and right in trait_cols:
                trait_cols[f"{left} AND {right}"] = trait_cols[left] & trait_cols[right]

    trait_df = pd.DataFrame(trait_cols, index=df_feat.index).astype(bool)
    return dedupe_columns(trait_df)


def score_negative_traits(trait_df: pd.DataFrame, y_hit: pd.Series) -> pd.DataFrame:
    target = y_hit.astype(int)
    rows: List[Dict[str, object]] = []
    total_n = int(len(target))
    base_rate = float(target.mean()) if total_n else 0.0

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
            "skip_gain_vs_base": base_rate - hit_rate_true,
            "zero_hit_trait": int(hits_true == 0 and support > 0),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["zero_hit_trait", "hit_rate_true", "support"], ascending=[False, True, False]).reset_index(drop=True)
    return out


def greedy_skip_bucket_search(trait_df: pd.DataFrame, y_hit: pd.Series, scored_negative_traits: pd.DataFrame, min_support: int = 50, top_k_traits: int = 40, max_depth: int = 4, max_hit_rate: float = 0.0015) -> pd.DataFrame:
    target = y_hit.astype(int)
    candidates = scored_negative_traits[(scored_negative_traits["support"] >= min_support) & (scored_negative_traits["hit_rate_true"] <= max_hit_rate)].head(top_k_traits)
    rows = []
    seen = set()

    for start_trait in candidates["trait"].tolist():
        mask = trait_df[start_trait].copy()
        chosen = [start_trait]

        for depth in range(1, max_depth + 1):
            support = int(mask.sum())
            if support < min_support:
                break

            hits = int(target[mask].sum())
            hit_rate = hits / support if support else 0.0
            base_rate = float(target.mean()) if len(target) else 0.0
            key = tuple(chosen)

            if key not in seen:
                seen.add(key)
                rows.append({
                    "bucket_id": 0,
                    "depth": len(chosen),
                    "bucket_traits": " AND ".join(chosen),
                    "support": support,
                    "hits": hits,
                    "misses": int(support - hits),
                    "hit_rate": hit_rate,
                    "base_rate_core025": base_rate,
                    "skip_gain_vs_base": base_rate - hit_rate,
                    "zero_hit_bucket": int(hits == 0 and support > 0),
                })

            if depth == max_depth:
                break

            best_next_trait = None
            best_next_rate = hit_rate
            best_next_support = support
            best_next_mask = None

            for nxt in candidates["trait"].tolist():
                if nxt in chosen:
                    continue
                nxt_mask = mask & trait_df[nxt]
                nxt_support = int(nxt_mask.sum())
                if nxt_support < min_support:
                    continue
                nxt_hits = int(target[nxt_mask].sum())
                nxt_rate = nxt_hits / nxt_support if nxt_support else 0.0

                if (nxt_rate < best_next_rate) or (np.isclose(nxt_rate, best_next_rate) and nxt_support > best_next_support):
                    best_next_rate = nxt_rate
                    best_next_support = nxt_support
                    best_next_trait = nxt
                    best_next_mask = nxt_mask

            if best_next_trait is None:
                break

            chosen.append(best_next_trait)
            mask = best_next_mask

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["zero_hit_bucket", "hit_rate", "support", "depth"], ascending=[False, True, False, True]).reset_index(drop=True)
        out["bucket_id"] = np.arange(1, len(out) + 1)
    return out


def bucket_mask_from_traits(trait_df: pd.DataFrame, bucket_traits: str) -> pd.Series:
    parts = [p.strip() for p in str(bucket_traits).split(" AND ") if p.strip()]
    if not parts:
        return pd.Series([False] * len(trait_df), index=trait_df.index)
    mask = pd.Series([True] * len(trait_df), index=trait_df.index)
    for t in parts:
        if t not in trait_df.columns:
            mask &= False
        else:
            mask &= trait_df[t].fillna(False).astype(bool)
    return mask


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
        due_pct = float(sum(current_gap >= gp for gp in gaps) / len(gaps)) if len(gaps) > 0 else 0.0
        recent_50_rate = float(g.tail(50)["next_is_core025_hit"].mean()) if len(g) > 0 else 0.0

        rows.append({
            "stream_id": stream_id,
            "jurisdiction": g.iloc[-1]["jurisdiction"],
            "game": g.iloc[-1]["game"],
            "events": n,
            "core025_hits": hits,
            "core025_hit_rate": base_rate,
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
    latest = latest[["stream_id", "jurisdiction", "game", "date_dt", "result4"]].rename(columns={"date_dt": "seed_date", "result4": "seed"})
    return dedupe_columns(latest.reset_index(drop=True))


def build_current_skip_table(current_seeds: pd.DataFrame, stream_profiles: pd.DataFrame, skip_buckets: pd.DataFrame, mine_level: str, top_skip_buckets_to_apply: int, skip_score_threshold: float) -> pd.DataFrame:
    merged = current_seeds.merge(stream_profiles, on=["stream_id", "jurisdiction", "game"], how="left")
    feat_rows = [compute_seed_features(seed) for seed in merged["seed"].astype(str)]
    df_feat = pd.DataFrame(feat_rows)
    df_feat = dedupe_columns(pd.concat([merged.reset_index(drop=True), df_feat.reset_index(drop=True)], axis=1))
    trait_df = build_trait_matrix(df_feat, mine_level=mine_level)

    selected_buckets = skip_buckets.head(int(top_skip_buckets_to_apply)).copy() if len(skip_buckets) else pd.DataFrame()
    fire_counts, fired_ids = [], []
    for i in trait_df.index:
        hit_ids = []
        for _, row in selected_buckets.iterrows():
            if bool(bucket_mask_from_traits(trait_df.loc[[i]], row["bucket_traits"]).iloc[0]):
                hit_ids.append(int(row["bucket_id"]))
        fire_counts.append(len(hit_ids))
        fired_ids.append(",".join(map(str, hit_ids)))

    df_feat["skip_bucket_fire_count"] = fire_counts
    df_feat["fired_skip_bucket_ids"] = fired_ids
    df_feat["skip_score"] = (
        0.40 * percentile_rank_series(1 - df_feat["core025_hit_rate"].fillna(0)) +
        0.20 * percentile_rank_series(1 - df_feat["recent_50_core025_rate"].fillna(0)) +
        0.40 * percentile_rank_series(df_feat["skip_bucket_fire_count"].fillna(0))
    ).clip(lower=0, upper=1)
    df_feat["skip_class"] = df_feat["skip_score"].apply(lambda s: "SKIP" if s >= float(skip_score_threshold) else "DO NOT AUTO-SKIP")

    keep_cols = [
        "stream_id", "jurisdiction", "game", "seed_date", "seed",
        "events", "core025_hits", "core025_hit_rate",
        "current_gap_since_last_hit", "recent_50_core025_rate",
        "skip_bucket_fire_count", "fired_skip_bucket_ids",
        "skip_score", "skip_class",
    ]
    out = df_feat[keep_cols].copy()
    out = out.sort_values(["skip_score", "skip_bucket_fire_count", "core025_hit_rate"], ascending=[False, False, True]).reset_index(drop=True)
    return dedupe_columns(out)


def build_true_walkforward_skip_backtest(model_df: pd.DataFrame, trait_df: pd.DataFrame, skip_buckets: pd.DataFrame, min_train_events_per_stream: int = 10, top_skip_buckets_to_apply: int = 10, skip_score_threshold: float = 0.60) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = model_df[["stream_id", "jurisdiction", "game", "event_date", "seed", "next_result4", "next_core025_member", "next_is_core025_hit"]].copy()
    base = dedupe_columns(base).sort_values(["event_date", "stream_id", "seed"]).reset_index(drop=True)

    stream_events_seen = defaultdict(int)
    stream_hits_seen = defaultdict(int)
    stream_recent50 = defaultdict(lambda: deque(maxlen=50))
    selected_buckets = skip_buckets.head(int(top_skip_buckets_to_apply)).copy() if len(skip_buckets) else pd.DataFrame()

    rows = []
    for i, r in base.iterrows():
        stream_id = r["stream_id"]
        events_seen = stream_events_seen[stream_id]
        hits_seen = stream_hits_seen[stream_id]
        stream_hit_rate = (hits_seen / events_seen) if events_seen > 0 else 0.0
        recent50_rate = float(np.mean(stream_recent50[stream_id])) if len(stream_recent50[stream_id]) > 0 else 0.0

        row_trait_mask = trait_df.loc[[i]]
        bucket_ids = []
        for _, b in selected_buckets.iterrows():
            if bool(bucket_mask_from_traits(row_trait_mask, b["bucket_traits"]).iloc[0]):
                bucket_ids.append(int(b["bucket_id"]))
        bucket_fire_count = len(bucket_ids)

        history_mix = min(events_seen / max(int(min_train_events_per_stream), 1), 1.0)
        skip_score = (
            0.40 * ((1 - stream_hit_rate) * history_mix + 0.5 * (1 - history_mix)) +
            0.20 * ((1 - recent50_rate) * history_mix + 0.5 * (1 - history_mix)) +
            0.40 * (bucket_fire_count / max(int(top_skip_buckets_to_apply), 1))
        )
        skip_score = float(max(0.0, min(1.0, skip_score)))
        skip_class = "SKIP" if skip_score >= float(skip_score_threshold) else "DO NOT AUTO-SKIP"

        rows.append({
            "event_date": r["event_date"],
            "stream_id": stream_id,
            "jurisdiction": r["jurisdiction"],
            "game": r["game"],
            "seed": r["seed"],
            "next_result4": r["next_result4"],
            "next_core025_member": r["next_core025_member"],
            "next_is_core025_hit": int(r["next_is_core025_hit"]),
            "train_events_seen_for_stream": int(events_seen),
            "train_hits_seen_for_stream": int(hits_seen),
            "wf_stream_hit_rate_before_event": float(stream_hit_rate),
            "wf_recent50_rate_before_event": float(recent50_rate),
            "skip_bucket_fire_count": int(bucket_fire_count),
            "fired_skip_bucket_ids": ",".join(map(str, bucket_ids)),
            "skip_score": skip_score,
            "skip_class": skip_class,
        })

        outcome = int(r["next_is_core025_hit"])
        stream_recent50[stream_id].append(outcome)
        stream_events_seen[stream_id] += 1
        stream_hits_seen[stream_id] += outcome

    wf = dedupe_columns(pd.DataFrame(rows))
    total_events = int(len(wf))
    total_hits = int(wf["next_is_core025_hit"].sum()) if len(wf) else 0

    class_rows = []
    for cls in ["SKIP", "DO NOT AUTO-SKIP"]:
        sub = wf[wf["skip_class"] == cls].copy() if len(wf) else pd.DataFrame()
        events = int(len(sub))
        hits = int(sub["next_is_core025_hit"].sum()) if len(sub) else 0
        class_rows.append({
            "skip_class": cls,
            "events": events,
            "events_pct": events / total_events if total_events else 0.0,
            "core025_hits": hits,
            "hit_rate": hits / events if events else 0.0,
            "hit_share_of_all_hits": hits / total_hits if total_hits else 0.0,
            "avg_skip_score": float(sub["skip_score"].mean()) if len(sub) else np.nan,
        })
    class_summary = pd.DataFrame(class_rows)

    plan_rows = []
    plans = {
        "Skip only SKIP": wf["skip_class"] != "SKIP" if len(wf) else pd.Series(dtype=bool),
        "Play everything": pd.Series([True] * len(wf), index=wf.index) if len(wf) else pd.Series(dtype=bool),
    }
    for plan_name, play_mask in plans.items():
        played = wf[play_mask].copy() if len(wf) else pd.DataFrame()
        skipped = wf[~play_mask].copy() if len(wf) else pd.DataFrame()
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
    return dedupe_columns(wf), dedupe_columns(class_summary), dedupe_columns(plan_summary)


def build_summary_text(raw_history: pd.DataFrame, transitions: pd.DataFrame, negative_traits: pd.DataFrame, skip_buckets: pd.DataFrame, current_skip: pd.DataFrame, wf_class_summary: pd.DataFrame, wf_plan_summary: pd.DataFrame) -> str:
    lines = []
    lines.append("CORE 025 SKIP ENGINE SUMMARY")
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append("")
    lines.append(f"Raw history rows: {len(raw_history):,}")
    lines.append(f"Transition events: {len(transitions):,}")
    lines.append(f"Core 025 family hits in transitions: {int(transitions['next_is_core025_hit'].sum()):,}")
    lines.append(f"Core 025 base rate in transitions: {float(transitions['next_is_core025_hit'].mean()):.4f}")
    lines.append("")
    lines.append("Top negative skip traits:")
    if len(negative_traits) == 0:
        lines.append("  - none")
    else:
        for _, r in negative_traits.head(15).iterrows():
            lines.append(f"  - {r['trait']} | support={int(r['support'])} | hit_rate_true={r['hit_rate_true']:.4f} | zero_hit={int(r['zero_hit_trait'])}")
    lines.append("")
    lines.append("Top stacked skip buckets:")
    if len(skip_buckets) == 0:
        lines.append("  - none")
    else:
        for _, r in skip_buckets.head(15).iterrows():
            lines.append(f"  - bucket_id={int(r['bucket_id'])} | support={int(r['support'])} | hit_rate={r['hit_rate']:.4f} | zero_hit_bucket={int(r['zero_hit_bucket'])} | {r['bucket_traits']}")
    lines.append("")
    lines.append("Current skip recommendations:")
    if len(current_skip) == 0:
        lines.append("  - none")
    else:
        for _, r in current_skip.head(15).iterrows():
            lines.append(f"  - {r['stream_id']} | seed={r['seed']} | skip_score={r['skip_score']:.3f} | class={r['skip_class']} | fired_buckets={r['fired_skip_bucket_ids']}")
    lines.append("")
    lines.append("TRUE WALK-FORWARD skip backtest by class:")
    for _, r in wf_class_summary.iterrows():
        lines.append(f"  - {r['skip_class']} | events={int(r['events'])} | hits={int(r['core025_hits'])} | hit_rate={r['hit_rate']:.4f} | hit_share={r['hit_share_of_all_hits']:.4f}")
    lines.append("")
    lines.append("TRUE WALK-FORWARD skip backtest play plans:")
    for _, r in wf_plan_summary.iterrows():
        lines.append(f"  - {r['plan']} | plays={int(r['plays'])} | plays_saved={int(r['plays_saved'])} | hits_kept={int(r['core025_hits_kept'])} | hits_skipped={int(r['core025_hits_skipped'])} | hit_retention={r['hit_retention_pct']:.4f} | hit_rate_on_played={r['hit_rate_on_played_events']:.4f}")
    return "\n".join(lines)


def run_skip_engine_pipeline(main_raw_df: pd.DataFrame, last24_raw_df: Optional[pd.DataFrame], mine_level: str, min_trait_support: int, bucket_min_support: int, bucket_top_k: int, bucket_max_depth: int, max_hit_rate_for_skip_trait: float, top_skip_buckets_to_apply: int, skip_score_threshold: float, min_train_events_per_stream: int) -> Dict[str, object]:
    main_history = prepare_raw_history(main_raw_df)
    last24_history = prepare_raw_history(last24_raw_df) if last24_raw_df is not None else None
    transitions = build_transition_events(main_history)
    seed_feat_rows = [compute_seed_features(seed) for seed in transitions["seed"].astype(str)]
    seed_feat = pd.DataFrame(seed_feat_rows)
    model_df = dedupe_columns(pd.concat([transitions.reset_index(drop=True), seed_feat.reset_index(drop=True)], axis=1))
    trait_df = build_trait_matrix(model_df, mine_level=mine_level)
    negative_traits = score_negative_traits(trait_df, model_df["next_is_core025_hit"].astype(int))
    negative_traits = negative_traits[negative_traits["support"] >= int(min_trait_support)].copy().reset_index(drop=True)

    skip_buckets = greedy_skip_bucket_search(
        trait_df=trait_df,
        y_hit=model_df["next_is_core025_hit"].astype(int),
        scored_negative_traits=negative_traits,
        min_support=int(bucket_min_support),
        top_k_traits=int(bucket_top_k),
        max_depth=int(bucket_max_depth),
        max_hit_rate=float(max_hit_rate_for_skip_trait),
    )

    stream_profiles = compute_stream_profiles(transitions)
    current_seeds = get_current_seed_rows(main_history, last24_history)
    current_skip = build_current_skip_table(
        current_seeds=current_seeds,
        stream_profiles=stream_profiles,
        skip_buckets=skip_buckets,
        mine_level=mine_level,
        top_skip_buckets_to_apply=int(top_skip_buckets_to_apply),
        skip_score_threshold=float(skip_score_threshold),
    )

    wf_events, wf_class_summary, wf_plan_summary = build_true_walkforward_skip_backtest(
        model_df=model_df,
        trait_df=trait_df,
        skip_buckets=skip_buckets,
        min_train_events_per_stream=int(min_train_events_per_stream),
        top_skip_buckets_to_apply=int(top_skip_buckets_to_apply),
        skip_score_threshold=float(skip_score_threshold),
    )

    summary_text = build_summary_text(
        raw_history=main_history,
        transitions=transitions,
        negative_traits=negative_traits,
        skip_buckets=skip_buckets,
        current_skip=current_skip,
        wf_class_summary=wf_class_summary,
        wf_plan_summary=wf_plan_summary,
    )

    return {
        "main_history": main_history,
        "last24_history": last24_history,
        "transitions": transitions,
        "model_df": model_df,
        "trait_df": trait_df,
        "negative_traits": negative_traits,
        "skip_buckets": skip_buckets,
        "stream_profiles": stream_profiles,
        "current_seeds": current_seeds,
        "current_skip": current_skip,
        "walkforward_events": wf_events,
        "walkforward_class_summary": wf_class_summary,
        "walkforward_plan_summary": wf_plan_summary,
        "summary_text": summary_text,
        "completed_at_utc": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
    }


def run_streamlit_app() -> None:
    st.set_page_config(page_title="Core 025 Skip Engine", layout="wide")
    st.title("Core 025 Skip Engine")
    st.caption("This app is built only to find historically weak Core 025 environments and stacked skip buckets. It does not try to predict the winning member.")

    with st.sidebar:
        st.header("Mining settings")
        mine_level = st.selectbox("Mine level", options=["standard", "expanded"], index=1)
        min_trait_support = st.number_input("Minimum trait support", min_value=3, value=12, step=1)
        bucket_min_support = st.number_input("Skip bucket minimum support", min_value=10, value=50, step=5)
        bucket_top_k = st.number_input("Skip bucket top K traits", min_value=5, value=40, step=5)
        bucket_max_depth = st.number_input("Skip bucket max depth", min_value=1, value=4, step=1)
        max_hit_rate_for_skip_trait = st.number_input("Max hit rate allowed for skip trait/bucket", min_value=0.0, max_value=0.05, value=0.0015, step=0.0005, format="%.4f")
        top_skip_buckets_to_apply = st.number_input("Top skip buckets to apply", min_value=1, value=10, step=1)
        skip_score_threshold = st.slider("Skip score threshold", min_value=0.0, max_value=1.0, value=0.60, step=0.01)
        min_train_events_per_stream = st.number_input("Walk-forward min train events per stream", min_value=1, value=10, step=1)
        top_rows = st.number_input("Rows to display per table", min_value=5, value=25, step=5)
        if st.button("Clear stored results"):
            st.session_state["skip_engine_results"] = None
            st.rerun()

    st.subheader("Upload files")
    main_file = st.file_uploader("Required main history file (full history)", type=["txt", "tsv", "csv", "xlsx", "xls"], key="main_history_uploader")
    last24_file = st.file_uploader("Optional last 24 file (same raw-history format)", type=["txt", "tsv", "csv", "xlsx", "xls"], key="last24_history_uploader")
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

    run_clicked = st.button("Run Core 025 Skip Engine", type="primary")
    if run_clicked:
        try:
            with st.spinner("Mining negative traits, building stacked skip buckets, scoring current streams, and running TRUE walk-forward skip backtest..."):
                results = run_skip_engine_pipeline(
                    main_raw_df=main_raw_df,
                    last24_raw_df=last24_raw_df,
                    mine_level=mine_level,
                    min_trait_support=int(min_trait_support),
                    bucket_min_support=int(bucket_min_support),
                    bucket_top_k=int(bucket_top_k),
                    bucket_max_depth=int(bucket_max_depth),
                    max_hit_rate_for_skip_trait=float(max_hit_rate_for_skip_trait),
                    top_skip_buckets_to_apply=int(top_skip_buckets_to_apply),
                    skip_score_threshold=float(skip_score_threshold),
                    min_train_events_per_stream=int(min_train_events_per_stream),
                )
            st.session_state["skip_engine_results"] = results
            st.rerun()
        except Exception as e:
            st.exception(e)
            return

    results = st.session_state.get("skip_engine_results")
    if results is None:
        st.info("Click 'Run Core 025 Skip Engine' after uploading your file(s).")
        return

    transitions_df = safe_result_df(results, "transitions")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw history rows", f"{len(safe_result_df(results, 'main_history')):,}")
    c2.metric("Transition events", f"{len(transitions_df):,}")
    c3.metric("Core 025 hits", f"{int(transitions_df['next_is_core025_hit'].sum()) if 'next_is_core025_hit' in transitions_df.columns and len(transitions_df) else 0:,}")
    base_rate = float(transitions_df["next_is_core025_hit"].mean()) if "next_is_core025_hit" in transitions_df.columns and len(transitions_df) else 0.0
    c4.metric("Core 025 base rate", f"{base_rate:.4f}")

    st.subheader("Summary")
    st.text_area("Summary text", str(results.get("summary_text", "")), height=420)
    st.download_button("Download summary TXT", data=str(results.get("summary_text", "")).encode("utf-8"), file_name="core025_skip_engine_summary__2026-03-25.txt", mime="text/plain", key="dl_summary_txt")

    st.markdown("## TRUE walk-forward skip backtest")
    wf_class = safe_result_df(results, "walkforward_class_summary", ["skip_class", "events", "events_pct", "core025_hits", "hit_rate", "hit_share_of_all_hits", "avg_skip_score"])
    wf_plan = safe_result_df(results, "walkforward_plan_summary", ["plan", "plays", "plays_pct_of_all_events", "plays_saved", "plays_saved_pct", "core025_hits_kept", "core025_hits_skipped", "hit_retention_pct", "hit_rate_on_played_events"])
    wf_events = safe_result_df(results, "walkforward_events")
    tab_bt1, tab_bt2, tab_bt3 = st.tabs(["Walk-forward class summary", "Walk-forward play plans", "Walk-forward event table"])
    with tab_bt1:
        st.dataframe(safe_display_df(wf_class, int(top_rows)), use_container_width=True)
        st.download_button("Download walk-forward class summary CSV", data=df_to_csv_bytes(wf_class), file_name="core025_skip_engine_walkforward_class_summary__2026-03-25.csv", mime="text/csv", key="dl_wf_class_summary_csv")
    with tab_bt2:
        st.dataframe(safe_display_df(wf_plan, int(top_rows)), use_container_width=True)
        st.download_button("Download walk-forward play plans CSV", data=df_to_csv_bytes(wf_plan), file_name="core025_skip_engine_walkforward_play_plans__2026-03-25.csv", mime="text/csv", key="dl_wf_plan_summary_csv")
    with tab_bt3:
        st.dataframe(safe_display_df(wf_events, int(top_rows)), use_container_width=True)
        st.download_button("Download walk-forward events CSV", data=df_to_csv_bytes(wf_events), file_name="core025_skip_engine_walkforward_events__2026-03-25.csv", mime="text/csv", key="dl_wf_events_csv")

    st.markdown("## Current skip recommendations")
    current_skip_df = safe_result_df(results, "current_skip")
    st.dataframe(safe_display_df(current_skip_df, int(top_rows)), use_container_width=True)
    st.download_button("Download current skip recommendations CSV", data=df_to_csv_bytes(current_skip_df), file_name="core025_current_skip_recommendations__2026-03-25.csv", mime="text/csv", key="dl_current_skip_csv")

    st.markdown("## Top negative skip traits")
    negative_traits_df = safe_result_df(results, "negative_traits")
    st.dataframe(safe_display_df(negative_traits_df, int(top_rows)), use_container_width=True)
    st.download_button("Download negative skip traits CSV", data=df_to_csv_bytes(negative_traits_df), file_name="core025_negative_skip_traits__2026-03-25.csv", mime="text/csv", key="dl_negative_traits_csv")

    st.markdown("## Stacked skip buckets")
    skip_buckets_df = safe_result_df(results, "skip_buckets")
    st.dataframe(safe_display_df(skip_buckets_df, int(top_rows)), use_container_width=True)
    st.download_button("Download skip buckets CSV", data=df_to_csv_bytes(skip_buckets_df), file_name="core025_skip_buckets__2026-03-25.csv", mime="text/csv", key="dl_skip_buckets_csv")

    st.markdown("## Transition event table")
    st.dataframe(safe_display_df(transitions_df, int(top_rows)), use_container_width=True)
    st.download_button("Download transition events CSV", data=df_to_csv_bytes(transitions_df), file_name="core025_transition_events__2026-03-25.csv", mime="text/csv", key="dl_transitions_csv")


def main():
    if has_streamlit_context():
        if "skip_engine_results" not in st.session_state:
            st.session_state["skip_engine_results"] = None
        run_streamlit_app()
    else:
        raise SystemExit("Run this file with: streamlit run core025_skip_engine_app_v2__2026-03-25.py")


if __name__ == "__main__":
    main()
