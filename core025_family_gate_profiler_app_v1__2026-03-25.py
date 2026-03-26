#!/usr/bin/env python3
# core025_combined_engine_v2__2026-03-25.py

import pandas as pd
import numpy as np
import re
import streamlit as st
from collections import Counter

CORE025_SET = {"0025","0225","0255"}

def norm(r):
    d = re.findall(r"\d", str(r))
    return "".join(d[:4]) if len(d)>=4 else None

def member(r):
    if r is None: return None
    s="".join(sorted(r))
    return s if s in CORE025_SET else None

def load(f):
    name=f.name.lower()
    if name.endswith(".csv"): return pd.read_csv(f)
    if name.endswith(".txt") or name.endswith(".tsv"):
        return pd.read_csv(f, sep="\t", header=None)
    if name.endswith(".xlsx"): return pd.read_excel(f)

def prep(df):
    if len(df.columns)==4:
        df.columns=["date","jurisdiction","game","result"]
    df["date"]=pd.to_datetime(df["date"], errors="coerce")
    df["r4"]=df["result"].apply(norm)
    df["member"]=df["r4"].apply(member)
    df["hit"]=df["member"].notna().astype(int)
    df["stream"]=df["jurisdiction"].astype(str)+"|"+df["game"].astype(str)
    return df.dropna(subset=["r4"]).reset_index(drop=True)

def feats(s):
    d=[int(x) for x in s]
    cnt=Counter(d)
    return {
        "sum":sum(d),
        "spread":max(d)-min(d),
        "even":sum(x%2==0 for x in d),
        "high":sum(x>=5 for x in d),
        "unique":len(cnt),
        "pair":int(len(cnt)<4),
        "max_rep":max(cnt.values())
    }

def build(df):
    rows=[]
    for s,g in df.groupby("stream"):
        g=g.sort_values("date").reset_index(drop=True)
        for i in range(1,len(g)):
            rows.append({
                "stream":s,
                "seed":g.loc[i-1,"r4"],
                "hit":g.loc[i,"hit"]
            })
    out=pd.DataFrame(rows)
    f=pd.DataFrame([feats(x) for x in out["seed"]])
    return pd.concat([out,f],axis=1)

def mine_neg(df):
    rows=[]
    for col in ["sum","spread","even","high","unique","pair","max_rep"]:
        for v in sorted(df[col].unique()):
            m=df[col]==v
            if m.sum()<20: continue
            hr=df.loc[m,"hit"].mean()
            rows.append((col,v,hr,m.sum()))
    return pd.DataFrame(rows, columns=["col","val","hit_rate","support"]).sort_values("hit_rate")

def mine_pos(df):
    base=df["hit"].mean()
    rows=[]
    for col in ["sum","spread","even","high","unique","pair","max_rep"]:
        for v in sorted(df[col].unique()):
            m=df[col]==v
            if m.sum()<20: continue
            hr=df.loc[m,"hit"].mean()
            if hr>base:
                rows.append((col,v,hr,m.sum()))
    return pd.DataFrame(rows, columns=["col","val","hit_rate","support"]).sort_values("hit_rate", ascending=False)

def score_row(f, neg, pos):
    skip=0
    gate=0

    for _,t in neg.head(15).iterrows():
        if f[t["col"]] == t["val"]:
            skip += 1

    for _,t in pos.head(15).iterrows():
        if f[t["col"]] == t["val"]:
            gate += 1

    score = gate - skip
    return score, skip, gate

def apply(df, neg, pos):
    classes=[]
    scores=[]

    for _,r in df.iterrows():
        f=feats(r["seed"])
        score, skip, gate = score_row(f, neg, pos)

        if score <= -1:
            cls="SKIP"
        elif score >= 1:
            cls="STRONG PLAY"
        else:
            cls="WEAK PLAY"

        classes.append(cls)
        scores.append(score)

    df["score"]=scores
    df["class"]=classes
    return df

def app():
    st.title("Combined Engine v2 (Scoring-Based)")

    file=st.file_uploader("Upload history file")
    if not file:
        return

    df=prep(load(file))
    tr=build(df)

    neg=mine_neg(tr)
    pos=mine_pos(tr)

    res=apply(tr,neg,pos)

    st.subheader("Class Counts")
    st.write(res["class"].value_counts())

    st.subheader("Hit Rates")
    st.write(res.groupby("class")["hit"].mean())

    st.subheader("Score Distribution")
    st.write(res["score"].value_counts().sort_index())

    st.subheader("Download Results")
    st.download_button("Download CSV", res.to_csv(index=False), "combined_v2_results.csv")

if __name__=="__main__":
    app()
