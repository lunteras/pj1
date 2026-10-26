# -*- coding: utf-8 -*-
"""보유상품수 × 납입액 분위 상호작용 확인 (ASCII 열 이름, 콘솔 인코딩 문제 회피)"""
import io, sys, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np

SRC = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"
fin = pd.read_csv(SRC, encoding="utf-8-sig")
ins = fin[fin["customer_type"] == "보험"].copy()
T = 2024
past = ins[ins["transaction_start_year"] <= T]
past_ids = set(past["customer_id"].unique())
fut_ids = set(ins.loc[ins["transaction_start_year"] > T, "customer_id"].unique())
cust = ins.drop_duplicates("customer_id").set_index("customer_id")
pool = cust[cust.index.isin(past_ids)].copy()
pool["label"] = pool.index.isin(fut_ids).astype(int)
prod = past.groupby("customer_id").agg(product_cnt=("product_name", "size"), pay_sum=("transaction_amt", "sum"))
pool = pool.join(prod)
print("pool", len(pool), "NaN pay", int(pool["pay_sum"].isna().sum()))
pool["pay_q"] = pd.qcut(pool["pay_sum"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
for n_g, sub in pool.groupby("product_cnt"):
    if len(sub) < 100:
        continue
    print(f"products {n_g}: n={len(sub):,} rate={sub['label'].mean()*100:.1f}%")
    gq = sub.groupby("pay_q")["label"].mean() * 100
    print("   ", {str(k): round(v, 1) for k, v in gq.items()})
