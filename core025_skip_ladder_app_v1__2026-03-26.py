#!/usr/bin/env python3
"""
core025_skip_ladder_app_v1__2026-03-26.py

Purpose
-------
Retention-targeted skip ladder for Core 025.

This app does one job:
- score every historical transition with a skip score
- rank events from strongest skip signal to weakest
- build a ladder showing how far into SKIP you can go
  while preserving a chosen minimum hit-retention target

This is meant to let you choose the most aggressive skip cutoff
that still keeps at least your required retention percentage.

Outputs
-------
- scored event table
- retention ladder table
- recommended cutoff at or above your target retention
- current stream scoring using optional last-24 file
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


def build_transition_events(history_df: pd.DataFrame) -> pd.DataFrame:
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


def digit_list(seed: str) -> List[int]:
    return [int(ch) for ch in str(seed)]


def feature_dict(seed: str) -> Dict[str, object]:
    d = digit_list(seed)
    cnt = Counter(d)
    s = sum(d)
    spread = max(d) - min(d)
    even = sum(x % 2 == 0 for x in d)
    high = sum(x >= 5 for x in d)
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
        "high": high,
        "unique": unique,
        "pair": int(unique < 4),
        "max_rep": max(cnt.values()),
        "pos1": d[0],
        "pos2": d[1],
        "pos3": d[2],
        "pos4": d[3],
        "consec_links": consec_links,
        "mirrorpair_cnt": mirrorpair_cnt,
    }
    for k in DIGITS:
        out[f"has{k}"] = int(k in cnt)
        out[f"cnt{k}"] = int(cnt.get(k, 0))
    return out


def build_feature_table(transitions_df: pd.DataFrame) -> pd.DataFrame:
    feats = [feature_dict(seed) for seed in transitions_df["seed"].astype(str)]
    feat_df = pd.DataFrame(feats)
    return dedupe_columns(pd.concat([transitions_df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1))


def mine_negative_traits(df: pd.DataFrame, min_support: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    base_rate = float(df["next_is_core025_hit"].mean())

    candidate_cols = ["sum", "spread", "even", "high", "unique", "pair", "max_rep", "pos1", "pos2", "pos3", "pos4", "consec_links", "mirrorpair_cnt"] + [f"has{k}" for k in DIGITS] + [f"cnt{k}" for k in DIGITS]

    for col in candidate_cols:
        vals = sorted(df[col].dropna().unique().tolist())
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
        out = out.sort_values(["zero_hit_trait", "hit_rate", "support"], ascending=[False, True, False]).reset_index(drop=True)
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


def build_skip_score_table(
    feat_df: pd.DataFrame,
    negative_traits_df: pd.DataFrame,
    top_negative_traits_to_use: int,
) -> pd.DataFrame:
    work = feat_df.copy()
    selected = negative_traits_df.head(int(top_negative_traits_to_use)).copy()

    fire_counts: List[int] = []
    fired_traits: List[str] = []

    trait_list = selected["trait"].tolist()

    for idx in work.index:
        row_df = work.loc[[idx]]
        fired: List[str] = []
        for t in trait_list:
            if bool(eval_single_trait(row_df, t).iloc[0]):
                fired.append(t)
        fire_counts.append(len(fired))
        fired_traits.append(" | ".join(fired))

    work["skip_fire_count"] = fire_counts
    work["fired_skip_traits"] = fired_traits

    # stronger skip score = more suppressive
    work["trait_fire_pct"] = percentile_rank_series(work["skip_fire_count"].fillna(0))
    work["stream_negative_pct"] = percentile_rank_series(1 - work.groupby("stream_id")["next_is_core025_hit"].transform("mean"))
    work["recent50_negative_pct"] = percentile_rank_series(1 - work["recent_50_hit_rate_before_event"].fillna(0))

    work["skip_score"] = (
        0.50 * work["trait_fire_pct"].fillna(0) +
        0.30 * work["stream_negative_pct"].fillna(0) +
        0.20 * work["recent50_negative_pct"].fillna(0)
    ).clip(lower=0, upper=1)

    return dedupe_columns(work)


def build_retention_ladder(
    scored_df: pd.DataFrame,
    rung_count: int,
) -> pd.DataFrame:
    df = scored_df.sort_values(["skip_score", "skip_fire_count"], ascending=[False, False]).reset_index(drop=True)
    total_events = len(df)
    total_hits = int(df["next_is_core025_hit"].sum())

    if total_events == 0:
        return pd.DataFrame()

    cutoffs = np.linspace(0, total_events, int(rung_count) + 1, dtype=int)[1:]
    rows: List[Dict[str, object]] = []

    for rank_cut in cutoffs:
        skip_mask = pd.Series([False] * total_events)
        skip_mask.iloc[:rank_cut] = True

        skipped = df[skip_mask]
        played = df[~skip_mask]

        plays_saved = int(len(skipped))
        hits_skipped = int(skipped["next_is_core025_hit"].sum()) if len(skipped) else 0
        hits_kept = int(played["next_is_core025_hit"].sum()) if len(played) else 0

        max_skip_score_included = float(skipped["skip_score"].min()) if len(skipped) else np.nan
        min_skip_score_not_included = float(played["skip_score"].max()) if len(played) else np.nan

        rows.append({
            "ladder_rank": len(rows) + 1,
            "events_marked_skip": plays_saved,
            "plays_saved_pct": plays_saved / total_events if total_events else 0.0,
            "hits_skipped": hits_skipped,
            "hits_kept": hits_kept,
            "hit_retention_pct": hits_kept / total_hits if total_hits else 0.0,
            "hit_rate_on_played_events": hits_kept / len(played) if len(played) else 0.0,
            "max_skip_score_included": max_skip_score_included,
            "next_score_after_cutoff": min_skip_score_not_included,
        })

    out = pd.DataFrame(rows)
    return dedupe_columns(out)


def recommend_cutoff(ladder_df: pd.DataFrame, target_retention_pct: float) -> pd.DataFrame:
    if len(ladder_df) == 0:
        return pd.DataFrame()
    ok = ladder_df[ladder_df["hit_retention_pct"] >= float(target_retention_pct)].copy()
    if len(ok) == 0:
        return ladder_df.head(1).copy()
    # choose most aggressive skip that still meets target = max plays_saved_pct
    best = ok.sort_values(["plays_saved_pct", "hit_rate_on_played_events"], ascending=[False, False]).head(1).copy()
    return dedupe_columns(best)


def current_seed_rows(history_df: pd.DataFrame, last24_history: Optional[pd.DataFrame]) -> pd.DataFrame:
    source = last24_history if last24_history is not None and len(last24_history) else history_df
    latest = source.sort_values(["stream_id", "date_dt", "file_order"], ascending=[True, True, False]).groupby("stream_id", as_index=False).tail(1).copy()
    feat_df = pd.DataFrame([feature_dict(x) for x in latest["result4"]])
    latest = latest.reset_index(drop=True)
    out = pd.concat([
        latest[["stream_id", "jurisdiction", "game", "date_dt", "result4"]].rename(columns={"date_dt": "seed_date", "result4": "seed"}),
        feat_df
    ], axis=1)
    return dedupe_columns(out)


def score_current_streams(
    current_df: pd.DataFrame,
    history_scored_df: pd.DataFrame,
    negative_traits_df: pd.DataFrame,
    top_negative_traits_to_use: int,
    chosen_skip_score_cutoff: float,
) -> pd.DataFrame:
    work = current_df.copy()
    selected = negative_traits_df.head(int(top_negative_traits_to_use)).copy()
    trait_list = selected["trait"].tolist()

    fire_counts: List[int] = []
    fired_traits: List[str] = []

    for idx in work.index:
        row_df = work.loc[[idx]]
        fired: List[str] = []
        for t in trait_list:
            if bool(eval_single_trait(row_df, t).iloc[0]):
                fired.append(t)
        fire_counts.append(len(fired))
        fired_traits.append(" | ".join(fired))

    work["skip_fire_count"] = fire_counts
    work["fired_skip_traits"] = fired_traits

    stream_hist = history_scored_df.groupby("stream_id")["next_is_core025_hit"].mean().rename("stream_hit_rate")
    stream_hist_recent = history_scored_df.groupby("stream_id")["recent_50_hit_rate_before_event"].mean().rename("stream_recent50")
    work = work.merge(stream_hist, on="stream_id", how="left")
    work = work.merge(stream_hist_recent, on="stream_id", how="left")

    work["trait_fire_pct"] = percentile_rank_series(work["skip_fire_count"].fillna(0))
    work["stream_negative_pct"] = percentile_rank_series(1 - work["stream_hit_rate"].fillna(history_scored_df["next_is_core025_hit"].mean()))
    work["recent50_negative_pct"] = percentile_rank_series(1 - work["stream_recent50"].fillna(history_scored_df["recent_50_hit_rate_before_event"].mean()))

    work["skip_score"] = (
        0.50 * work["trait_fire_pct"].fillna(0) +
        0.30 * work["stream_negative_pct"].fillna(0) +
        0.20 * work["recent50_negative_pct"].fillna(0)
    ).clip(lower=0, upper=1)

    work["skip_class"] = np.where(work["skip_score"] >= float(chosen_skip_score_cutoff), "SKIP", "PLAY")
    out = work[["stream_id", "jurisdiction", "game", "seed_date", "seed", "skip_fire_count", "fired_skip_traits", "skip_score", "skip_class"]].copy()
    out = out.sort_values(["skip_score", "skip_fire_count"], ascending=[False, False]).reset_index(drop=True)
    return dedupe_columns(out)


def build_summary_text(
    transitions_df: pd.DataFrame,
    negative_traits_df: pd.DataFrame,
    ladder_df: pd.DataFrame,
    recommended_df: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("CORE 025 SKIP LADDER SUMMARY")
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append("")
    lines.append(f"Transition events: {len(transitions_df):,}")
    lines.append(f"Core025 hits: {int(transitions_df['next_is_core025_hit'].sum()):,}")
    lines.append(f"Core025 base rate: {float(transitions_df['next_is_core025_hit'].mean()):.4f}")
    lines.append("")
    lines.append("Top negative traits:")
    for _, r in negative_traits_df.head(10).iterrows():
        lines.append(f"  - {r['trait']} | support={int(r['support'])} | hit_rate={r['hit_rate']:.4f} | gain={r['gain_vs_base']:.4f}")
    lines.append("")
    if len(recommended_df):
        r = recommended_df.iloc[0]
        lines.append("Recommended cutoff at/above retention target:")
        lines.append(f"  - plays_saved_pct={r['plays_saved_pct']:.4f}")
        lines.append(f"  - hit_retention_pct={r['hit_retention_pct']:.4f}")
        lines.append(f"  - hit_rate_on_played_events={r['hit_rate_on_played_events']:.4f}")
        lines.append(f"  - skip_score_cutoff={r['max_skip_score_included']:.6f}")
    return "\n".join(lines)


def run_pipeline(
    main_raw_df: pd.DataFrame,
    last24_raw_df: Optional[pd.DataFrame],
    min_trait_support: int,
    top_negative_traits_to_use: int,
    rung_count: int,
    target_retention_pct: float,
) -> Dict[str, object]:
    main_history = prepare_history(main_raw_df)
    last24_history = prepare_history(last24_raw_df) if last24_raw_df is not None else None

    transitions_df = build_transition_events(main_history)
    feat_df = build_feature_table(transitions_df)
    negative_traits_df = mine_negative_traits(feat_df, min_support=int(min_trait_support))
    scored_df = build_skip_score_table(
        feat_df=feat_df,
        negative_traits_df=negative_traits_df,
        top_negative_traits_to_use=int(top_negative_traits_to_use),
    )
    ladder_df = build_retention_ladder(scored_df, rung_count=int(rung_count))
    recommended_df = recommend_cutoff(ladder_df, target_retention_pct=float(target_retention_pct))

    chosen_cutoff = float(recommended_df.iloc[0]["max_skip_score_included"]) if len(recommended_df) else 1.0
    current_df = current_seed_rows(main_history, last24_history)
    current_scored_df = score_current_streams(
        current_df=current_df,
        history_scored_df=scored_df,
        negative_traits_df=negative_traits_df,
        top_negative_traits_to_use=int(top_negative_traits_to_use),
        chosen_skip_score_cutoff=chosen_cutoff,
    )

    summary_text = build_summary_text(
        transitions_df=transitions_df,
        negative_traits_df=negative_traits_df,
        ladder_df=ladder_df,
        recommended_df=recommended_df,
    )

    return {
        "main_history": main_history,
        "last24_history": last24_history,
        "transitions": transitions_df,
        "features": feat_df,
        "negative_traits": negative_traits_df,
        "scored_events": scored_df,
        "retention_ladder": ladder_df,
        "recommended_cutoff": recommended_df,
        "current_scored_streams": current_scored_df,
        "summary_text": summary_text,
        "completed_at_utc": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
    }


def run_streamlit_app() -> None:
    st.set_page_config(page_title="Core025 Skip Ladder", layout="wide")

    if "skip_ladder_results" not in st.session_state:
        st.session_state["skip_ladder_results"] = None

    st.title("Core025 Skip Ladder v1")
    st.caption("Retention-targeted skip ranking system. Choose how far into SKIP you can go while preserving your minimum retention target.")

    with st.sidebar:
        st.header("Controls")
        min_trait_support = st.number_input("Minimum trait support", min_value=3, value=12, step=1)
        top_negative_traits_to_use = st.number_input("Top negative traits to use for scoring", min_value=1, value=15, step=1)
        rung_count = st.number_input("Ladder rung count", min_value=5, value=50, step=5)
        target_retention_pct = st.slider("Target hit retention", min_value=0.50, max_value=0.99, value=0.75, step=0.01)
        rows_to_show = st.number_input("Rows to display", min_value=5, value=25, step=5)

        if st.button("Clear stored results"):
            st.session_state["skip_ladder_results"] = None
            st.rerun()

    st.subheader("Upload files")
    main_file = st.file_uploader(
        "Required main history file (full history)",
        type=["txt", "tsv", "csv", "xlsx", "xls"],
        key="skip_ladder_main_uploader",
    )
    last24_file = st.file_uploader(
        "Optional last 24 file (same raw-history format)",
        type=["txt", "tsv", "csv", "xlsx", "xls"],
        key="skip_ladder_last24_uploader",
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

    if st.button("Run Core025 Skip Ladder", type="primary"):
        try:
            with st.spinner("Scoring skip ladder and computing retention cutoffs..."):
                results = run_pipeline(
                    main_raw_df=main_raw_df,
                    last24_raw_df=last24_raw_df,
                    min_trait_support=int(min_trait_support),
                    top_negative_traits_to_use=int(top_negative_traits_to_use),
                    rung_count=int(rung_count),
                    target_retention_pct=float(target_retention_pct),
                )
            st.session_state["skip_ladder_results"] = results
            st.rerun()
        except Exception as e:
            st.exception(e)
            return

    results = st.session_state.get("skip_ladder_results")
    if results is None:
        st.info("Click the run button after uploading the main history file.")
        return

    st.success(f"Completed at UTC: {results['completed_at_utc']}")

    transitions_df = results["transitions"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transition events", f"{len(transitions_df):,}")
    c2.metric("Core025 hits", f"{int(transitions_df['next_is_core025_hit'].sum()):,}")
    c3.metric("Base rate", f"{float(transitions_df['next_is_core025_hit'].mean()):.4f}")
    c4.metric("Top traits used", f"{int(top_negative_traits_to_use):,}")

    st.subheader("Summary")
    st.text_area("Summary text", results["summary_text"], height=380)
    st.download_button(
        "Download summary TXT",
        data=results["summary_text"].encode("utf-8"),
        file_name="core025_skip_ladder_summary__2026-03-26.txt",
        mime="text/plain",
    )

    st.markdown("## Recommended cutoff")
    st.dataframe(results["recommended_cutoff"], use_container_width=True)
    st.download_button(
        "Download recommended cutoff CSV",
        data=df_to_csv_bytes(results["recommended_cutoff"]),
        file_name="core025_skip_ladder_recommended_cutoff__2026-03-26.csv",
        mime="text/csv",
    )

    st.markdown("## Retention ladder")
    st.dataframe(safe_display_df(results["retention_ladder"], int(rows_to_show)), use_container_width=True)
    st.download_button(
        "Download retention ladder CSV",
        data=df_to_csv_bytes(results["retention_ladder"]),
        file_name="core025_skip_ladder_retention_ladder__2026-03-26.csv",
        mime="text/csv",
    )

    st.markdown("## Current scored streams")
    st.dataframe(safe_display_df(results["current_scored_streams"], int(rows_to_show)), use_container_width=True)
    st.download_button(
        "Download current scored streams CSV",
        data=df_to_csv_bytes(results["current_scored_streams"]),
        file_name="core025_skip_ladder_current_scored_streams__2026-03-26.csv",
        mime="text/csv",
    )

    st.markdown("## Top negative traits")
    st.dataframe(safe_display_df(results["negative_traits"], int(rows_to_show)), use_container_width=True)
    st.download_button(
        "Download negative traits CSV",
        data=df_to_csv_bytes(results["negative_traits"]),
        file_name="core025_skip_ladder_negative_traits__2026-03-26.csv",
        mime="text/csv",
    )

    st.markdown("## Scored events")
    st.dataframe(safe_display_df(results["scored_events"], int(rows_to_show)), use_container_width=True)
    st.download_button(
        "Download scored events CSV",
        data=df_to_csv_bytes(results["scored_events"]),
        file_name="core025_skip_ladder_scored_events__2026-03-26.csv",
        mime="text/csv",
    )


def main():
    if has_streamlit_context():
        run_streamlit_app()
    else:
        raise SystemExit("Run this file with: streamlit run core025_skip_ladder_app_v1__2026-03-26.py")


if __name__ == "__main__":
    main()
