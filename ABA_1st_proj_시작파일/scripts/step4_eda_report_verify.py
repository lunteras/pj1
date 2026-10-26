# -*- coding: utf-8 -*-
"""EDA 리포트에 들어갈 분포 수치를 실데이터로 계산 (5-2 표의 숫자 대조용)"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
import pandas as pd
import numpy as np

SRC = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"
fin = pd.read_csv(SRC, encoding="utf-8-sig")
ins = fin[fin["customer_type"] == "보험"]
pool = ins.drop_duplicates("customer_id")
past = ins[ins["transaction_start_year"] <= 2024]
pool = pool[pool["customer_id"].isin(past["customer_id"])].copy()
n = len(pool)
print("pool:", n)

for col in ["age", "gender", "marital_status", "income_bracket", "region", "risk_grade", "credit_rating"]:
    vc = pool[col].value_counts(dropna=False)
    print(f"\n[{col}]")
    for v, c in vc.items():
        print(f"  {v}: {c} ({c/n*100:.1f}%)")

fut = ins[ins["transaction_start_year"] > 2024]
y = pool["customer_id"].isin(set(fut["customer_id"]))
print("\n[타겟]", y.sum(), f"({y.mean()*100:.1f}%)", "비", f"1:{round((1-y.mean())/y.mean(),1)}")
