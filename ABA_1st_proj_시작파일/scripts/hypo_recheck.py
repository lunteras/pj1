# -*- coding: utf-8 -*-
"""T=2023년 말 예측 프레임 — 가설 4개 재검증 (타겟: 2024~2025년 신규 상품 가입 여부)"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd
import numpy as np

SRC = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"
OUT = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\output"

from matplotlib import font_manager, pyplot as plt
names = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

fin = pd.read_csv(SRC, encoding="utf-8-sig")
ins = fin[fin["customer_type"] == "보험"].copy()
T = 2023

past = ins[ins["transaction_start_year"] <= T]
fut = ins[ins["transaction_start_year"] > T]
past_ids = set(past["customer_id"].unique())
fut_ids = set(fut["customer_id"].unique())

cust = ins.drop_duplicates("customer_id").set_index("customer_id")
pool = cust[cust.index.isin(past_ids)].copy()
pool["label"] = pool.index.isin(fut_ids).astype(int)

# T까지 알 수 있는 파생 값
first_year = past.groupby("customer_id")["transaction_start_year"].min()
pool["tenure"] = T - pool.index.map(first_year)
pool["보유상품수"] = past.groupby("customer_id").size()
pool["납입액합계"] = past.groupby("customer_id")["transaction_amt"].sum() / 1e4  # 만원

def cover_type(name):
    if "변액" in str(name): return "변액"
    if "연금" in str(name): return "기타"
    if str(name).strip() == "기타 상품": return "기타불명"
    return "보장"
past["cover_type"] = past["product_name"].map(cover_type)
past["갱신형"] = past["product_name"].str.contains("갱신형", na=False)

ct = past.groupby("customer_id")["cover_type"].agg(set)
pool["보장보유"] = ct.map(lambda s: "보장" in s).reindex(pool.index).fillna(False)
gang = past[past["갱신형"]].groupby("customer_id").size()
pool["갱신형보유"] = pool.index.isin(set(gang.index))

def rate_table(sub, col, order=None, label=""):
    g = sub.groupby(col)["label"].agg(["count", "sum", "mean"])
    g.columns = ["고객 수", "가입자", "가입률(%)"]
    g["가입률(%)"] = (g["가입자"] / g["고객 수"] * 100).round(1)
    if order: g = g.loc[order]
    print(f"\n[{label}]")
    print(g.to_string())
    small = g[g["고객 수"] < 30]
    if len(small): print(f"  ※ 30 미만 셀: {list(small.index)}")
    return g

print(f"=== 프레임 ===\n풀(T=2023년 말 기존 고객): {len(pool):,} / 양성(2024~25 신규 가입): {int(pool['label'].sum()):,} / 가입률: {pool['label'].mean()*100:.1f}%")

# ---- H0: 40~50대 & 소득 상위 2구간 프로필 ----
print("\n" + "="*60 + "\nH0. 40~50대 & 소득 상위 2구간 고객의 추가 가입률이 높다?")
pool["프로필"] = np.where((pool["age"].isin(["40대", "50대"])) &
                       (pool["income_bracket"].isin(["8000만원미만", "10000만원이상"])),
                       "충족", "미충족")
rate_table(pool, "프로필", ["충족", "미충족"], "H0-1 프로필 충족 여부")
rate_table(pool, "age", ["20대", "30대", "40대", "50대", "60대", "70대 이상"], "H0-2 연령대 단독")
rate_table(pool, "income_bracket", ["3000만원미만", "5000만원미만", "8000만원미만", "10000만원이상"], "H0-3 소득 단독")

# ---- H1: 신용 900점대이상 vs 800점대 이하 1.5배 ----
print("\n" + "="*60 + "\nH1. 900점대이상의 가입률은 800점대 이하의 1.5배 이상?")
credit_sub = pool[pool["credit_rating"] != "기타"].copy()
rate_table(credit_sub, "credit_rating", ["900점대이상", "800점대", "700점대", "300점대"], "H1-1 신용 점수대별")
rate_table(credit_sub, "risk_grade", ["A", "B", "C", "D", "E"], "H1-2 위험등급별")
print("\n[H1-3 연령 보정 — 연령대별 900점대이상 vs 이하 가입률(%)]")
credit_sub["신용고저"] = np.where(credit_sub["credit_rating"] == "900점대이상", "900이상", "800이하")
ct2 = pd.crosstab(credit_sub["age"], credit_sub["신용고저"], values=credit_sub["label"], aggfunc="mean") * 100
n2 = pd.crosstab(credit_sub["age"], credit_sub["신용고저"])
print(ct2.round(1).to_string())
print("셀 크기(최소):", int(n2.values.min()))

# ---- H2: 가입 후 경과 기간 ----
print("\n" + "="*60 + "\nH2. 가입 후 경과 기간이 길수록 가입률이 높다?")
pool["경과기간"] = pd.cut(pool["tenure"], [-1, 1, 4, 9, 100], labels=["0~1년", "2~4년", "5~9년", "10년 이상"])
rate_table(pool, "경과기간", ["0~1년", "2~4년", "5~9년", "10년 이상"], "H2 경과 기간별")

# ---- H3: 갱신형 vs 비갱신 ----
print("\n" + "="*60 + "\nH3. 갱신형 보유자의 가입률은 비갱신보다 낮다?")
bg = pool[pool["보장보유"]].copy()
bg["상품유형"] = np.where(bg["갱신형보유"], "갱신형 보유", "비갱신만")
rate_table(bg, "상품유형", ["갱신형 보유", "비갱신만"], "H3 보장 보유자 내 갱신형 여부")

# ---- 보너스: 보유 상품 수 ----
pool["보유규모"] = np.where(pool["보유상품수"] >= 2, "2개 이상", "1개")
rate_table(pool, "보유규모", ["1개", "2개 이상"], "보너스 T시점 보유 상품 수별")
print(f"보유 상품 수 분포: {pool['보유상품수'].value_counts().sort_index().to_dict()}")

# ---- 그림: 4패널 ----
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
BASE = pool["label"].mean() * 100
panels = [
    ("H0 프로필 충족 여부", pool.groupby("프로필")["label"].mean() * 100, ["충족", "미충족"]),
    ("H1 신용 점수대별", credit_sub.groupby("credit_rating")["label"].mean() * 100, ["900점대이상", "800점대", "700점대", "300점대"]),
    ("H2 가입 후 경과 기간별", pool.groupby("경과기간", observed=True)["label"].mean() * 100, ["0~1년", "2~4년", "5~9년", "10년 이상"]),
    ("H3 갱신형 보유 여부", bg.groupby("상품유형")["label"].mean() * 100, ["갱신형 보유", "비갱신만"]),
]
for ax, (title, s, order) in zip(axes.flat, panels):
    s = s.reindex(order)
    ax.bar([str(x) for x in s.index], s.values, color="#4C72B0")
    ax.axhline(BASE, color="crimson", ls="--", lw=1)
    ax.set_title(f"{title}\n(빨간 점선 = 전체 {BASE:.1f}%)", fontsize=10)
    ax.set_ylabel("2년 내 추가 가입률(%)")
    for i, v in enumerate(s.values):
        ax.text(i, v + 0.12, f"{v:.1f}", ha="center", fontsize=9)
fig.suptitle("T=2023년 말 기준 2년 내 추가 가입률 — 가설별 재검증 (풀 8,964명)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{OUT}\\예측_T2023_가설재검증.png", dpi=150)
plt.close(fig)
print("\n그림 저장: output/예측_T2023_가설재검증.png")
