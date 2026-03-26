#!/usr/bin/env python3
"""
core025_combined_engine_v3__2026-03-25.py

Combined Engine v3 (Adaptive Thresholds)
- Uses the same skip/gate trait idea as v2
- Adds sidebar controls for adaptive thresholds
- Shows class counts, hit rates, score distribution, and play-plan summaries
- Exports the full scored dataset

No placeholders. Full file.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

import pandas as pd
import numpy as np
import streamlit as st

CORE025_SET = {"0025", "0225", "0255"}


def norm_result(r: object) -> Optional[str]:
    d = re.findall(r"\d", str(r))
    return "".join(d[:4]) if len(d) >= 4 else None


def to_member(r4: Optional[str]) -> Optional[str]:
    if r4 is None:
        return None
    s = "".join(sorted(r4))
    return s if s in CORE025_SET else None


def load_table(uploaded_file) -> pd.DataFrame:
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
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if len(df.columns) == 4:
        df.columns = ["date", "jurisdiction", "game", "result"]
    else:
        cols = [str(c).lower() for c in df.columns]
        df.columns = cols
        rename_map = {}
        if "result" not in cols:
            for c in cols:
                if "result" in c:
                    rename_map[c] = "result"
                    break
        if "date" not in cols:
            for c in cols:
                if "date" in c:
                    rename_map[c] = "date"
                    break
        if "jurisdiction" not in cols:
            for c in cols:
                if "jurisdiction" in c or "state" in c:
                    rename_map[c] = "jurisdiction"
                    break
        if "game" not in cols:
            for c in cols:
                if "game" in c or "stream" in c:
                    rename_map[c] = "game"
                    break
        df = df.rename(columns=rename_map)
        needed = {"date", "jurisdiction", "game", "result"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["r4"] = df["result"].apply(norm_result)
    df["member"] = df["r4"].apply(to_member)
    df["hit"] = df["member"].notna().astype(int)
    df["stream"] = df["jurisdiction"].astype(str) + "|" + df["game"].astype(str)
    df = df.dropna(subset=["r4"]).reset_index(drop=True)
    return df


def seed_features(seed: str) -> Dict[str, int]:
    d = [int(x) for x in seed]
    cnt = Counter(d)
    return {
        "sum": sum(d),
        "spread": max(d) - min(d),
        "even": sum(x % 2 == 0 for x in d),
        "high": sum(x >= 5 for x in d),
        "unique": len(cnt),
        "pair": int(len(cnt) < 4),
        "max_rep": max(cnt.values()),
        "pos1": d[0],
        "pos2": d[1],
        "pos3": d[2],
        "pos4": d[3],
    }


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for stream, g in df.groupby("stream"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < 2:
            continue
        for i in range(1, len(g)):
            feat = seed_features(g.loc[i - 1, "r4"])
            rows.append({
                "stream": stream,
                "jurisdiction": g.loc[i, "jurisdiction"],
                "game": g.loc[i, "game"],
                "seed": g.loc[i - 1, "r4"],
                "event_date": g.loc[i, "date"],
                "hit": g.loc[i, "hit"],
                **feat,
            })
    return pd.DataFrame(rows)


def mine_negative(df: pd.DataFrame, min_support: int) -> pd.DataFrame:
    rows = []
    for col in ["sum", "spread", "even", "high", "unique", "pair", "max_rep", "pos1", "pos2", "pos3", "pos4"]:
        for v in sorted(df[col].dropna().unique()):
            m = df[col] == v
            support = int(m.sum())
            if support < min_support:
                continue
            hr = float(df.loc[m, "hit"].mean())
            rows.append({"col": col, "val": v, "hit_rate": hr, "support": support})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["hit_rate", "support"], ascending=[True, False]).reset_index(drop=True)
    return out


def mine_positive(df: pd.DataFrame, min_support: int) -> pd.DataFrame:
    base = float(df["hit"].mean())
    rows = []
    for col in ["sum", "spread", "even", "high", "unique", "pair", "max_rep", "pos1", "pos2", "pos3", "pos4"]:
        for v in sorted(df[col].dropna().unique()):
            m = df[col] == v
            support = int(m.sum())
            if support < min_support:
                continue
            hr = float(df.loc[m, "hit"].mean())
            if hr > base:
                rows.append({"col": col, "val": v, "hit_rate": hr, "support": support})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["hit_rate", "support"], ascending=[False, False]).reset_index(drop=True)
    return out


def score_row(feat: Dict[str, int], neg: pd.DataFrame, pos: pd.DataFrame, neg_top_n: int, pos_top_n: int) -> tuple[int, int, int]:
    skip = 0
    gate = 0
    for _, t in neg.head(neg_top_n).iterrows():
        if feat[t["col"]] == t["val"]:
            skip += 1
    for _, t in pos.head(pos_top_n).iterrows():
        if feat[t["col"]] == t["val"]:
            gate += 1
    score = gate - skip
    return score, skip, gate


def apply_engine(
    df: pd.DataFrame,
    neg: pd.DataFrame,
    pos: pd.DataFrame,
    neg_top_n: int,
    pos_top_n: int,
    skip_threshold: int,
    strong_threshold: int,
) -> pd.DataFrame:
    out = df.copy()
    classes = []
    scores = []
    skip_counts = []
    gate_counts = []

    for _, r in out.iterrows():
        feat = {
            "sum": r["sum"],
            "spread": r["spread"],
            "even": r["even"],
            "high": r["high"],
            "unique": r["unique"],
            "pair": r["pair"],
            "max_rep": r["max_rep"],
            "pos1": r["pos1"],
            "pos2": r["pos2"],
            "pos3": r["pos3"],
            "pos4": r["pos4"],
        }
        score, skip, gate = score_row(feat, neg, pos, neg_top_n=neg_top_n, pos_top_n=pos_top_n)
        if score <= skip_threshold:
            cls = "SKIP"
        elif score >= strong_threshold:
            cls = "STRONG PLAY"
        else:
            cls = "WEAK PLAY"
        scores.append(score)
        skip_counts.append(skip)
        gate_counts.append(gate)
        classes.append(cls)

    out["score"] = scores
    out["skip_count"] = skip_counts
    out["gate_count"] = gate_counts
    out["class"] = classes
    return out


def summarize_counts(df: pd.DataFrame) -> pd.DataFrame:
    vc = df["class"].value_counts(dropna=False).rename_axis("class").reset_index(name="count")
    vc["pct"] = vc["count"] / len(df)
    return vc


def summarize_hit_rates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("class", dropna=False)["hit"].agg(["count", "sum", "mean"]).reset_index()
    out = out.rename(columns={"sum": "hits", "mean": "hit_rate"})
    return out


def summarize_play_plan(df: pd.DataFrame) -> pd.DataFrame:
    total_events = len(df)
    total_hits = int(df["hit"].sum())

    def row_for_plan(name: str, play_mask: pd.Series) -> Dict[str, object]:
        played = df[play_mask]
        not_played = df[~play_mask]
        plays = len(played)
        hits_kept = int(played["hit"].sum())
        hits_not_played = int(not_played["hit"].sum())
        return {
            "plan": name,
            "plays": plays,
            "plays_pct_of_all_events": plays / total_events if total_events else 0.0,
            "plays_saved": total_events - plays,
            "plays_saved_pct": (total_events - plays) / total_events if total_events else 0.0,
            "core025_hits_kept": hits_kept,
            "core025_hits_not_played": hits_not_played,
            "hit_retention_pct": hits_kept / total_hits if total_hits else 0.0,
            "hit_rate_on_played_events": hits_kept / plays if plays else 0.0,
        }

    rows = [
        row_for_plan("Play STRONG only", df["class"] == "STRONG PLAY"),
        row_for_plan("Play STRONG + WEAK", df["class"].isin(["STRONG PLAY", "WEAK PLAY"])),
        row_for_plan("Play everything", pd.Series([True] * len(df), index=df.index)),
    ]
    return pd.DataFrame(rows)


def current_seed_rows(history_df: pd.DataFrame) -> pd.DataFrame:
    latest = history_df.sort_values("date").groupby("stream", as_index=False).tail(1).copy()
    feat_df = pd.DataFrame([seed_features(x) for x in latest["r4"]])
    latest = latest.reset_index(drop=True)
    out = pd.concat([latest[["stream", "jurisdiction", "game", "date", "r4"]].rename(columns={"date": "seed_date", "r4": "seed"}), feat_df], axis=1)
    return out


def score_current(
    current_df: pd.DataFrame,
    neg: pd.DataFrame,
    pos: pd.DataFrame,
    neg_top_n: int,
    pos_top_n: int,
    skip_threshold: int,
    strong_threshold: int,
) -> pd.DataFrame:
    rows = []
    for _, r in current_df.iterrows():
        feat = {
            "sum": r["sum"],
            "spread": r["spread"],
            "even": r["even"],
            "high": r["high"],
            "unique": r["unique"],
            "pair": r["pair"],
            "max_rep": r["max_rep"],
            "pos1": r["pos1"],
            "pos2": r["pos2"],
            "pos3": r["pos3"],
            "pos4": r["pos4"],
        }
        score, skip, gate = score_row(feat, neg, pos, neg_top_n=neg_top_n, pos_top_n=pos_top_n)
        if score <= skip_threshold:
            cls = "SKIP"
        elif score >= strong_threshold:
            cls = "STRONG PLAY"
        else:
            cls = "WEAK PLAY"
        rows.append({
            "stream": r["stream"],
            "jurisdiction": r["jurisdiction"],
            "game": r["game"],
            "seed_date": r["seed_date"],
            "seed": r["seed"],
            "score": score,
            "skip_count": skip,
            "gate_count": gate,
            "class": cls,
        })
    out = pd.DataFrame(rows).sort_values(["score", "gate_count", "skip_count"], ascending=[False, False, True]).reset_index(drop=True)
    return out


def run_app():
    st.set_page_config(page_title="Combined Engine v3", layout="wide")
    st.title("Combined Engine v3 (Adaptive Thresholds)")
    st.caption("Combined Skip + Gate engine with adaptive threshold controls.")

    with st.sidebar:
        st.header("Controls")
        min_support = st.number_input("Minimum trait support", min_value=3, value=20, step=1)
        neg_top_n = st.number_input("Top negative traits to use", min_value=1, value=15, step=1)
        pos_top_n = st.number_input("Top positive traits to use", min_value=1, value=15, step=1)
        skip_threshold = st.number_input("Skip threshold (score ≤ this => SKIP)", value=0, step=1)
        strong_threshold = st.number_input("Strong threshold (score ≥ this => STRONG)", value=2, step=1)
        rows_to_show = st.number_input("Rows to display", min_value=5, value=25, step=5)

    file = st.file_uploader("Upload history file", type=["csv", "txt", "tsv", "xlsx", "xls"])
    if not file:
        return

    try:
        hist = prepare_history(load_table(file))
    except Exception as e:
        st.error(f"Could not load file: {e}")
        return

    transitions = build_transitions(hist)
    neg = mine_negative(transitions, min_support=int(min_support))
    pos = mine_positive(transitions, min_support=int(min_support))
    scored = apply_engine(
        transitions,
        neg,
        pos,
        neg_top_n=int(neg_top_n),
        pos_top_n=int(pos_top_n),
        skip_threshold=int(skip_threshold),
        strong_threshold=int(strong_threshold),
    )

    class_counts = summarize_counts(scored)
    hit_rates = summarize_hit_rates(scored)
    play_plans = summarize_play_plan(scored)

    current = current_seed_rows(hist)
    current_scored = score_current(
        current,
        neg,
        pos,
        neg_top_n=int(neg_top_n),
        pos_top_n=int(pos_top_n),
        skip_threshold=int(skip_threshold),
        strong_threshold=int(strong_threshold),
    )

    st.subheader("Class Counts")
    st.dataframe(class_counts, use_container_width=True)

    st.subheader("Hit Rates by Class")
    st.dataframe(hit_rates, use_container_width=True)

    st.subheader("Play Plans")
    st.dataframe(play_plans, use_container_width=True)

    st.subheader("Score Distribution")
    score_dist = scored["score"].value_counts().sort_index().rename_axis("score").reset_index(name="count")
    st.dataframe(score_dist, use_container_width=True)

    st.subheader("Current Scored Streams")
    st.dataframe(current_scored.head(int(rows_to_show)), use_container_width=True)

    st.subheader("Top Negative Traits")
    st.dataframe(neg.head(int(rows_to_show)), use_container_width=True)

    st.subheader("Top Positive Traits")
    st.dataframe(pos.head(int(rows_to_show)), use_container_width=True)

    st.download_button("Download combined v3 results CSV", scored.to_csv(index=False), "combined_v3_results.csv", "text/csv")
    st.download_button("Download class counts CSV", class_counts.to_csv(index=False), "combined_v3_class_counts.csv", "text/csv")
    st.download_button("Download hit rates CSV", hit_rates.to_csv(index=False), "combined_v3_hit_rates.csv", "text/csv")
    st.download_button("Download play plans CSV", play_plans.to_csv(index=False), "combined_v3_play_plans.csv", "text/csv")
    st.download_button("Download current scored streams CSV", current_scored.to_csv(index=False), "combined_v3_current_scored_streams.csv", "text/csv")


if __name__ == "__main__":
    run_app()
