# -*- coding: utf-8 -*-
"""다음 가설 후보 탐색 — 아직 미확인 축의 신호 크기 훑기 (T=2024)"""
import io, sys, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

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
print(f"풀 {len(pool):,} / 양성 {int(pool['label'].sum()):,} / {pool['label'].mean()*100:.1f}%")

# 1) 상품 분류 축: 주보험 이름이 담고 있는 것들
prod = past.groupby("customer_id").agg(
    상품수=("product_name", "size"),
    이름=("product_name", lambda s: " | ".join(sorted(set(s)))),
    키워드=("transaction_keyword", lambda s: " | ".join(sorted(set(map(str, s))))),
    총납입=("transaction_amt", "sum"),
)
p2 = pool.join(prod)
print("\n[상품 이름 분해] T 시점 보유자의 상품 이름 토큰 훑기 (토큰별 보유자 수, 가입률)")
toks = {}
for idx, row in prod.iterrows():
    for t in set(str(row["이름"]).split(" | ")):
        toks.setdefault(t, []).append(idx)
rows = []
for t, ids in toks.items():
    if len(ids) < 200:  # 너무 희소한 토큰은 제외
        continue
    r = pool.loc[ids]
    rows.append((t, len(ids), r["label"].mean() * 100))
df_tok = pd.DataFrame(rows, columns=["토큰", "보유자", "가입률(%)"]).sort_values("가입률(%)", ascending=False)
print(df_tok.round(1).to_string(index=False))

# 2) transaction_keyword 토큰
print("\n[거래 키워드 토큰]")
toks2 = {}
for idx, row in prod.iterrows():
    for t in set(str(row["키워드"]).split(" | ")):
        toks2.setdefault(t, []).append(idx)
rows = []
for t, ids in toks2.items():
    if len(ids) < 300:
        continue
    r = pool.loc[ids]
    rows.append((t, len(ids), r["label"].mean() * 100))
print(pd.DataFrame(rows, columns=["토큰", "보유자", "가입률(%)"]).sort_values("가입률(%)", ascending=False).round(1).to_string(index=False))

# 3) 납입액 구간 (단독 축) — 가설 A와 다른 '금액' 축
pool["납입액합계"] = prod["총납입"].reindex(pool.index)
p3 = pool.copy()
p3["납입분위"] = pd.qcut(p3["납입액합계"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
g = p3.groupby("납입분위")["label"].agg(["count", "mean"])
g.columns = ["고객 수", "가입률(%)"]
g["가입률(%)"] = (g["가입률(%)"] * 100).round(1)
print("\n[납입액 5분위별 가입률]")
print(g.to_string())

# 4) 갱신형 vs 일반형 다시 — 상품 이름 기반 (보유상품수 조각에서)
print("\n[보유상품수 조각 안에서의 납입액분위 상호작용]")
for n_g, sub in p3.groupby("상품수"):
    if len(sub) < 100:
        continue
    q = sub["label"].mean() * 100
    print(f"  상품 {n_g}개: {len(sub):,}명 가입률 {q:.1f}%")
    gq = sub.groupby("납입분위")["label"].mean() * 100
    print("    납입분위별:", {k: round(v, 1) for k, v in gq.items()})
