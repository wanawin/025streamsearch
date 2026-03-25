#!/usr/bin/env python3
# core025_combined_engine_v1__2026-03-25.py

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
    if name.endswith(".txt") or name.endswith(".tsv"): return pd.read_csv(f, sep="\t", header=None)
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
    return {
        "sum":sum(d),
        "spread":max(d)-min(d),
        "even":sum(x%2==0 for x in d),
        "high":sum(x>=5 for x in d),
        "unique":len(set(d))
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
    for col in ["sum","spread","even","high","unique"]:
        for v in sorted(df[col].unique()):
            m=df[col]==v
            if m.sum()<20: continue
            hr=df.loc[m,"hit"].mean()
            rows.append((f"{col}={v}",hr,m.sum()))
    return pd.DataFrame(rows, columns=["trait","hit_rate","support"]).sort_values("hit_rate")

def mine_pos(df):
    base=df["hit"].mean()
    rows=[]
    for col in ["sum","spread","even","high","unique"]:
        for v in sorted(df[col].unique()):
            m=df[col]==v
            if m.sum()<20: continue
            hr=df.loc[m,"hit"].mean()
            if hr>base:
                rows.append((f"{col}={v}",hr,m.sum()))
    return pd.DataFrame(rows, columns=["trait","hit_rate","support"]).sort_values("hit_rate", ascending=False)

def apply(df, neg, pos):
    out=[]
    for _,r in df.iterrows():
        s=r["seed"]
        f=feats(s)
        skip=0
        gate=0
        for _,t in neg.head(10).iterrows():
            col,val=t["trait"].split("=")
            if str(f[col])==val:
                skip+=1
        for _,t in pos.head(10).iterrows():
            col,val=t["trait"].split("=")
            if str(f[col])==val:
                gate+=1
        if skip>=2:
            cls="SKIP"
        elif gate>=2:
            cls="STRONG PLAY"
        else:
            cls="WEAK PLAY"
        out.append(cls)
    df["class"]=out
    return df

def app():
    st.title("Combined Engine v1 (Skip + Gate)")
    file=st.file_uploader("Upload history")
    if not file: return
    df=prep(load(file))
    tr=build(df)
    neg=mine_neg(tr)
    pos=mine_pos(tr)
    res=apply(tr,neg,pos)

    st.subheader("Class Counts")
    st.write(res["class"].value_counts())

    st.subheader("Hit Rates")
    st.write(res.groupby("class")["hit"].mean())

    st.subheader("Download")
    st.download_button("Download results", res.to_csv(index=False), "combined_results.csv")

if __name__=="__main__":
    app()
