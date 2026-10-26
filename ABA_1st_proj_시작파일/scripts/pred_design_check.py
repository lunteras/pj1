# -*- coding: utf-8 -*-
"""예측 문제 설계 검증 — 시간 순 분할(예측 시점 T) 구조가 실제로 만들어지는지 확인"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd
import numpy as np

SRC = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"
fin = pd.read_csv(SRC, encoding="utf-8-sig")
ins = fin[fin["customer_type"] == "보험"].copy()

# ---- 1. 상품명 → 담보 유형 분류 규칙 ----
def cover_type(name):
    if "변액" in str(name):
        return "변액"
    if "연금" in str(name):
        return "기타"          # 연금저축 등 (변액연금은 위에서 먼저 변액으로 분류됨)
    if str(name).strip() == "기타 상품":
        return "기타불명"
    return "보장"

ins["cover_type"] = ins["product_name"].map(cover_type)
print("=== 행 단위 담보 유형 분류 결과 ===")
print(ins["cover_type"].value_counts())
for t in ["변액", "기타", "보장"]:
    top = ins.loc[ins["cover_type"] == t, "product_name"].value_counts().head(3)
    print(f"\n[{t} 상위 상품]")
    print(top.to_string())

# ---- 2. 재구성 담보 구성 vs 실제 cross_coverage 일치율 ----
def compose(types):
    has = set(types)
    order = [x for x in ["보장", "변액", "기타"] if x in has]  # 기타불명은 기타 취급 안 함(보수적)
    has2 = set("기타" if x == "기타불명" else x for x in has)
    order = [x for x in ["보장", "변액", "기타"] if x in has2]
    if not order:
        return "없음"
    if len(order) == 1:
        return order[0] + "only" if order[0] != "보장" else "보장only"
    return "+".join(order)

recon = ins.groupby("customer_id")["cover_type"].agg(compose)
cc = ins.drop_duplicates("customer_id").set_index("customer_id")["cross_coverage"]
cmp = pd.DataFrame({"recon": recon, "cc": cc}).reindex(cc.index)
match = (cmp["recon"] == cmp["cc"]).mean()
print(f"\n=== 재구성 담보 구성 vs cross_coverage 일치율: {match*100:.1f}% ===")
tab = pd.crosstab(cmp["recon"], cmp["cc"])
print(tab.to_string())

# ---- 3. 예측 시점 T별 — 풀(모수)과 양성(실제 추가 가입자) 수 ----
print("\n=== 예측 시점 T별 구조 (연도 단위) ===")
rows = []
for T in [2021, 2022, 2023, 2024]:
    past = ins[ins["transaction_start_year"] <= T]
    fut = ins[ins["transaction_start_year"] > T]
    past_ids = set(past["customer_id"].unique())
    fut_ids = set(fut["customer_id"].unique())
    fut_nonbg = set(fut.loc[fut["cover_type"] != "보장", "customer_id"].unique())
    # T 시점 보장only: 과거에 보유한 상품이 전부 보장 유형
    past_types = past.groupby("customer_id")["cover_type"].agg(set)
    bgonly_ids = set(past_types[past_types.map(lambda s: s == {"보장"})].index)
    rows.append({
        "예측 시점 T": f"{T}년 말",
        "기존 고객(풀)": f"{len(past_ids):,}",
        "T시점 보장only": f"{len(bgonly_ids):,}",
        "이후 추가가입(보장only 중)": f"{len(bgonly_ids & fut_ids):,}",
        "이후 담보확장(보장only 중)": f"{len(bgonly_ids & fut_nonbg):,}",
        "기존고객 전체 중 추가가입": f"{len(past_ids & fut_ids):,}",
    })
t_df = pd.DataFrame(rows)
print(t_df.to_string(index=False))

# ---- 4. 보장only-at-T 풀의 속성 요약 (T=2023 예시) — 모델링 표본 확인용 ----
T = 2023
past = ins[ins["transaction_start_year"] <= T]
past_types = past.groupby("customer_id")["cover_type"].agg(set)
bgonly_ids = past_types[past_types.map(lambda s: s == {"보장"})].index
cust = ins.drop_duplicates("customer_id").set_index("customer_id")
pool = cust.loc[cust.index.isin(bgonly_ids)]
fut = ins[ins["transaction_start_year"] > T]
fut_ids = set(fut["customer_id"].unique())
pool = pool.copy()
pool["label"] = pool.index.isin(fut_ids).astype(int)
print(f"\n=== T=2023년 말 보장only 풀 {len(pool):,}명 — 양성 {pool['label'].sum():,}명 ({pool['label'].mean()*100:.1f}%) ===")
print("연령대 분포:")
print(pool["age"].value_counts().to_string())
print("\n라벨별 연령대 비율(%):")
print((pd.crosstab(pool["age"], pool["label"], normalize="index") * 100).round(1).to_string())
