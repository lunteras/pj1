# -*- coding: utf-8 -*-
"""T=2024년 말 버전 — 재앙커링된 가설 4개 재검증 (타겟: 2025년 신규 상품 가입 여부, 지평 1년)"""
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
T = 2024

past = ins[ins["transaction_start_year"] <= T].copy()
fut_ids = set(ins.loc[ins["transaction_start_year"] > T, "customer_id"].unique())
past_ids = set(past["customer_id"].unique())

cust = ins.drop_duplicates("customer_id").set_index("customer_id")
pool = cust[cust.index.isin(past_ids)].copy()
pool["label"] = pool.index.isin(fut_ids).astype(int)
first_year = past.groupby("customer_id")["transaction_start_year"].min()
pool["tenure"] = T - pool.index.map(first_year)
pool["보유상품수"] = past.groupby("customer_id").size()
pool["납입액"] = past.groupby("customer_id")["transaction_amt"].sum() / 1e4  # 만원

def cover_type(name):
    if "변액" in str(name): return "변액"
    if "연금" in str(name): return "기타"
    if str(name).strip() == "기타 상품": return "기타불명"
    return "보장"
past["cover_type"] = past["product_name"].map(cover_type)
past["갱신형"] = past["product_name"].str.contains("갱신형", na=False)
ct = past.groupby("customer_id")["cover_type"].agg(set)
pool["보장보유"] = ct.map(lambda s: "보장" in s).reindex(pool.index).fillna(False)
gang_ids = set(past.loc[past["갱신형"], "customer_id"].unique())
pool["갱신형보유"] = pool.index.isin(gang_ids)

BASE = pool["label"].mean() * 100
print(f"프레임: 풀 {len(pool):,}명 / 양성(2025년 가입) {int(pool['label'].sum()):,}명 / 전체 가입률 {BASE:.1f}%")

def rt(sub, col, order, label):
    g = sub.groupby(col, observed=False)["label"].agg(["count", "sum", "mean"])
    g.columns = ["고객 수", "가입자", "가입률(%)"]
    g["가입률(%)"] = (g["가입자"] / g["고객 수"] * 100).round(1)
    g = g.loc[order]
    print(f"\n[{label}]")
    print(g.to_string())
    small = g[g["고객 수"] < 30]
    if len(small): print(f"  ※ 고객 수 30 미만 셀: {list(small.index)}")
    tiny = g[g["가입자"] < 10]
    if len(tiny): print(f"  ※ 가입자 10 미만 셀(비율 불안정): {list(tiny.index)}")
    return g

# ================= H0 =================
print("\n" + "#"*62 + "\nH0. 40~50대 & 소득 상위 2구간 고객의 1년 내 추가 가입률이 높다?")
pool["프로필"] = np.where(pool["age"].isin(["40대", "50대"]) &
                       pool["income_bracket"].isin(["8000만원미만", "10000만원이상"]), "충족", "미충족")
g0 = rt(pool, "프로필", ["충족", "미충족"], "H0-1 프로필 충족 여부")
rt(pool, "age", ["20대", "30대", "40대", "50대", "60대", "70대 이상"], "H0-2 연령대 단독")
rt(pool, "income_bracket", ["3000만원미만", "5000만원미만", "8000만원미만", "10000만원이상"], "H0-3 소득 단독")

# ================= H1 =================
print("\n" + "#"*62 + "\nH1. 900점대이상 고객의 가입률은 800점대 이하의 1.5배 이상?")
credit_sub = pool[pool["credit_rating"] != "기타"].copy()
g1 = rt(credit_sub, "credit_rating", ["900점대이상", "800점대", "700점대", "300점대"], "H1-1 신용 점수대별")
rt(credit_sub, "risk_grade", ["A", "B", "C", "D", "E"], "H1-2 위험등급별")

# ================= H2 =================
print("\n" + "#"*62 + "\nH2. 가입 후 경과 기간이 길수록 가입률이 높다?")
pool["경과기간"] = pd.cut(pool["tenure"], [-1, 1, 4, 9, 100], labels=["0~1년", "2~4년", "5~9년", "10년 이상"])
g2 = rt(pool, "경과기간", ["0~1년", "2~4년", "5~9년", "10년 이상"], "H2-1 경과 기간별")
print("\n[H2 검증 — 경과 기간별 갱신형 보유 비율]")
h2v = pool.groupby("경과기간", observed=False).agg(고객수=("label", "count"), 가입자=("label", "sum"),
                                            갱신형보유비율=("갱신형보유", "mean"))
h2v["갱신형보유비율(%)"] = (h2v["갱신형보유비율"] * 100).round(1)
h2v["가입률(%)"] = (h2v["가입자"] / h2v["고객수"] * 100).round(1)
print(h2v[["고객수", "갱신형보유비율(%)", "가입률(%)"]].to_string())
# 2024년 가입자만 따로 (T시점 경과 0년)
new24 = pool[pool["tenure"] == 0]
print(f"\n2024년 가입자(T시점 경과 0년): {len(new24):,}명 / 가입률 {new24['label'].mean()*100:.1f}%")

# ================= H3 =================
print("\n" + "#"*62 + "\nH3. 갱신형 보유자의 가입률은 비갱신보다 낮다?")
bg = pool[pool["보장보유"]].copy()
bg["상품유형"] = np.where(bg["갱신형보유"], "갱신형 보유", "비갱신만")
g3 = rt(bg, "상품유형", ["갱신형 보유", "비갱신만"], "H3-1 갱신형 여부별")
print("\n[H3 검증 — 상품유형별 고객 프로필]")
h3v = bg.groupby("상품유형", observed=False).agg(고객수=("label", "count"), 가입자=("label", "sum"),
                                          중장년비율=("age", lambda s: s.isin(["50대", "60대", "70대 이상"]).mean()),
                                          납입액중앙값만원=("납입액", "median"), 보유상품수평균=("보유상품수", "mean"))
h3v["가입률(%)"] = (h3v["가입자"] / h3v["고객수"] * 100).round(1)
h3v["중장년비율(%)"] = (h3v["중장년비율"] * 100).round(1)
h3v["납입액중앙값만원"] = h3v["납입액중앙값만원"].round(0)
print(h3v.to_string())

# ================= 보너스 =================
print("\n" + "#"*62 + "\n보너스. T시점 보유 상품 수")
pool["보유규모"] = np.where(pool["보유상품수"] >= 2, "2개 이상", "1개")
g4 = rt(pool, "보유규모", ["1개", "2개 이상"], "보너스 보유 상품 수별")

# ================= 그림 4패널 =================
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
panels = [
    ("H0 프로필 충족 여부", g0), ("H1 신용 점수대별", g1),
    ("H2 가입 후 경과 기간별", g2), ("H3 갱신형 보유 여부", g3),
]
for ax, (title, g) in zip(axes.flat, panels):
    labels = [f"{i}\n(n={int(n):,})" for i, n in zip(g.index, g["고객 수"])]
    ax.bar(labels, g["가입률(%)"], color="#4C72B0")
    ax.axhline(BASE, color="crimson", ls="--", lw=1)
    for i, v in enumerate(g["가입률(%)"]):
        ax.text(i, v + 0.08, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_title(f"{title}\n(빨간 점선 = 전체 {BASE:.1f}%)", fontsize=10)
    ax.set_ylabel("1년 내 추가 가입률(%)")
    ax.tick_params(axis="x", labelsize=8)
fig.suptitle("T=2024년 말 기준 2025년 추가 가입률 — 가설별 재검증 (풀 38,932명)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{OUT}\\예측_T2024_가설재검증.png", dpi=150)
plt.close(fig)
print("\n그림 저장: output/예측_T2024_가설재검증.png")
