# -*- coding: utf-8 -*-
# 납입액 중앙값 재확인 — T=2024 프레임 + 대용라벨 프레임
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
import pandas as pd, numpy as np

df = pd.read_csv(r"C:\Users\user\AppData\Local\Temp\helpycode\finance\소비자.csv", encoding="utf-8-sig")
ins = df[df["customer_type"] == "보험"].copy()

# 고객별 납입액 합계
pays = ins.groupby("customer_id")["transaction_amt"].sum()

# (1) T=2024 프레임
pool_ids = ins[ins["transaction_start_year"] <= 2024]["customer_id"].unique()
pool = ins[ins["customer_id"].isin(pool_ids)].drop_duplicates("customer_id")
later = set(ins[ins["transaction_start_year"] > 2024]["customer_id"])
pool["target"] = pool["customer_id"].isin(later).astype(int)
pool = pool.merge(pays.rename("pay"), left_on="customer_id", right_index=True, how="left")
pool = pool.merge(ins.groupby("customer_id").size().rename("cnt"), left_on="customer_id", right_index=True, how="left")
print("== T=2024 프레임 (풀", len(pool), "· 양성", pool["target"].sum(), ") ==")
print("납입액 합계 중앙값: 전체", pool["pay"].median(),
      "/ 1개 보유", pool.loc[pool["cnt"] == 1, "pay"].median(),
      "/ 2개 이상", pool.loc[pool["cnt"] > 1, "pay"].median())
print("타겟별: 미가입", pool.loc[pool["target"] == 0, "pay"].median(),
      "/ 가입", pool.loc[pool["target"] == 1, "pay"].median())

# (2) 대용라벨 프레임 (보장 포함 고객): 확장자(보장+변액/기타) vs 보장only — cross_coverage가 이미 결합 문자열
allc = ins.drop_duplicates("customer_id")[["cross_coverage", "customer_id"]].copy()
g = allc[allc["cross_coverage"].str.contains("보장", na=False)].copy()
g["pay"] = g["customer_id"].map(pays)
is_only = g["cross_coverage"].eq("보장only")
exp = g[~is_only]; only = g[is_only]
print("== 대용라벨 프레임 (보장 포함", len(g), ") ==")
print("확장자", len(exp), "· 중앙값", round(exp["pay"].median()),
      "/ 보장only", len(only), "· 중앙값", round(only["pay"].median()))
