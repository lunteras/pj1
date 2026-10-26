# -*- coding: utf-8 -*-
"""피처 확정 전 마지막 확인 — 병력·성별 (T=2023 / T=2024 실제 타겟)"""
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
    pool["병력3"] = np.where(pool["disease_history"] == "없음", "없음",
                       np.where(pool["disease_history"] == "기타", "기타", "있음"))
    print(f"\n===== T={T} (전체 가입률 {pool['label'].mean()*100:.1f}%) =====")
    for c, order in [("병력3", ["없음", "있음", "기타"]), ("gender", ["여", "남"])]:
        g = pool.groupby(c)["label"].agg(["count", "sum", "mean"])
        g.columns = ["고객 수", "가입자", "가입률(%)"]
        g["가입률(%)"] = (g["가입자"] / g["고객 수"] * 100).round(1)
        print(f"[{c}] {g.loc[order].to_dict('index')}")
