#!/usr/bin/env python3
# core025_skip_engine_app_v3__2026-03-25.py
# Deep separation version (no placeholders)

from __future__ import annotations
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict, deque
from datetime import datetime

import streamlit as st

CORE025_SET = {"0025","0225","0255"}

def percentile_rank_series(s):
    return s.rank(method="average", pct=True)

def normalize_result(r):
    d = re.findall(r"\d", str(r))
    return "".join(d[:4]) if len(d)>=4 else None

def to_member(r):
    if r is None: return None
    s = "".join(sorted(r))
    return s if s in CORE025_SET else None

def load_df(file):
    name = file.name.lower()
    if name.endswith(".csv"): return pd.read_csv(file)
    if name.endswith(".txt") or name.endswith(".tsv"):
        return pd.read_csv(file, sep="\t", header=None)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    raise ValueError("Unsupported")

def prep(df):
    if len(df.columns)==4:
        df.columns=["date","jurisdiction","game","result"]
    else:
        df.columns=[c.lower() for c in df.columns]
    df["date"]=pd.to_datetime(df["date"], errors="coerce")
    df["r4"]=df["result"].apply(normalize_result)
    df["member"]=df["r4"].apply(to_member)
    df["is_hit"]=df["member"].notna().astype(int)
    df["stream"]=df["jurisdiction"].astype(str)+"|"+df["game"].astype(str)
    df=df.dropna(subset=["r4"]).reset_index(drop=True)
    return df

def transitions(df):
    rows=[]
    for s,g in df.sort_values("date").groupby("stream"):
        g=g.reset_index(drop=True)
        for i in range(1,len(g)):
            rows.append({
                "stream":s,
                "seed":g.loc[i-1,"r4"],
                "hit":g.loc[i,"is_hit"],
                "idx":i
            })
    return pd.DataFrame(rows)

def seed_feats(seed):
    d=[int(x) for x in seed]
    return {
        "sum":sum(d),
        "spread":max(d)-min(d),
        "even":sum(x%2==0 for x in d),
        "odd":sum(x%2 for x in d),
        "high":sum(x>=5 for x in d),
        "low":sum(x<=4 for x in d),
        "unique":len(set(d)),
        "pair":int(len(set(d))<4),
        "pos1":d[0],
        "pos4":d[3],
        "sum_mod3":sum(d)%3,
        "sum_mod4":sum(d)%4,
    }

def build_feats(df):
    feats=[seed_feats(x) for x in df["seed"]]
    fdf=pd.DataFrame(feats)
    return pd.concat([df.reset_index(drop=True),fdf],axis=1)

def mine_negative(df, min_support=20):
    rows=[]
    base=df["hit"].mean()
    for col in ["sum","spread","even","odd","high","low","unique","pair","sum_mod3","sum_mod4"]:
        for val in sorted(df[col].unique()):
            m=df[col]==val
            sup=m.sum()
            if sup<min_support: continue
            hr=df.loc[m,"hit"].mean()
            rows.append({
                "trait":f"{col}={val}",
                "support":sup,
                "hit_rate":hr,
                "gain":base-hr
            })
    out=pd.DataFrame(rows)
    return out.sort_values(["hit_rate","support"],ascending=[True,False])

def build_buckets(df, traits, depth=3):
    buckets=[]
    for i,row in traits.head(30).iterrows():
        t=row["trait"]
        mask=df.eval(t.replace("=","=="))
        cur_mask=mask.copy()
        used=[t]
        for d in range(depth):
            sup=cur_mask.sum()
            if sup<20: break
            hr=df.loc[cur_mask,"hit"].mean()
            buckets.append({
                "traits":" & ".join(used),
                "support":sup,
                "hit_rate":hr
            })
            best=None
            for _,r2 in traits.head(30).iterrows():
                t2=r2["trait"]
                if t2 in used: continue
                m2=cur_mask & df.eval(t2.replace("=","=="))
                if m2.sum()<20: continue
                hr2=df.loc[m2,"hit"].mean()
                if best is None or hr2<best[1]:
                    best=(t2,hr2,m2)
            if best is None: break
            used.append(best[0])
            cur_mask=best[2]
    return pd.DataFrame(buckets).sort_values(["hit_rate","support"])

def current_skip(df, buckets):
    rows=[]
    for i,r in df.iterrows():
        fires=0
        for _,b in buckets.head(10).iterrows():
            try:
                if eval(b["traits"],{},r.to_dict()):
                    fires+=1
            except: pass
        score=fires/10
        rows.append({
            "stream":r["stream"],
            "seed":r["seed"],
            "skip_score":score,
            "class":"SKIP" if score>=0.5 else "PLAY"
        })
    return pd.DataFrame(rows)

def app():
    st.title("Core025 Skip Engine v3 (Deep)")
    main=st.file_uploader("Main history file")
    if not main:
        return
    df=prep(load_df(main))
    tr=transitions(df)
    trf=build_feats(tr)
    neg=mine_negative(trf)
    buckets=build_buckets(trf,neg)
    cur=current_skip(trf.tail(200),buckets)

    st.subheader("Top Negative Traits")
    st.dataframe(neg.head(25))

    st.subheader("Buckets")
    st.dataframe(buckets.head(25))

    st.subheader("Current Skip")
    st.dataframe(cur)

    st.download_button("Download buckets", buckets.to_csv(index=False), "skip_buckets_v3.csv")

if __name__=="__main__":
    app()
