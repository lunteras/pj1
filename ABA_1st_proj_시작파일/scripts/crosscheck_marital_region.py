# -*- coding: utf-8 -*-
"""EDA 재실행에서 나온 신호 2개의 교차 확인 — 결혼·지역 (T=2023 vs T=2024)"""
import io, sys, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

SRC = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"
fin = pd.read_csv(SRC, encoding="utf-8-sig")
ins = fin[fin["customer_type"] == "보험"].copy()
cust = ins.drop_duplicates("customer_id").set_index("customer_id")

for T in [2023, 2024]:
    past = ins[ins["transaction_start_year"] <= T]
    past_ids = set(past["customer_id"].unique())
    fut_ids = set(ins.loc[ins["transaction_start_year"] > T, "customer_id"].unique())
    pool = cust[cust.index.isin(past_ids)].copy()
    pool["label"] = pool.index.isin(fut_ids).astype(int)
    print(f"\n===== T={T} (풀 {len(pool):,} / 양성 {int(pool['label'].sum()):,} / {pool['label'].mean()*100:.1f}%) =====")

    g = pool.groupby("marital_status")["label"].agg(["count", "sum", "mean"])
    g.columns = ["고객 수", "가입자", "가입률(%)"]
    g["가입률(%)"] = (g["가입자"] / g["고객 수"] * 100).round(1)
    print("[결혼]", g.to_dict("index"))

    r = pool.groupby("region")["label"].agg(["count", "sum", "mean"])
    r.columns = ["고객 수", "가입자", "가입률(%)"]
    r["가입률(%)"] = (r["가입자"] / r["고객 수"] * 100).round(1)
    print("[지역]", r["가입률(%)"].sort_values(ascending=False).to_string())

    # 결혼 효과가 연령의 그림자인지 — 연령대별 기혼/미혼 가입률
    ct = pd.crosstab(pool["age"], pool["marital_status"], values=pool["label"], aggfunc="mean") * 100
    n = pd.crosstab(pool["age"], pool["marital_status"])
    print("[연령대별 기혼/미혼 가입률(%)]")
    print(ct.round(1).to_string())
    print("  셀 크기 최소:", int(n.values.min()))
