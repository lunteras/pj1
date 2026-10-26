# -*- coding: utf-8 -*-
"""T=2024 점검 — 상관관계 히트맵 그림 (숫자형 피어슨 + 범주형 크라메르 V)"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

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
ins = fin[fin["customer_type"] == "보험"]
T = 2024
past = ins[ins["transaction_start_year"] <= T]
cust = ins.drop_duplicates("customer_id").set_index("customer_id")
past_ids = set(past["customer_id"].unique())
pool = cust[cust.index.isin(past_ids)].copy()
pool["label"] = pool.index.isin(set(ins.loc[ins["transaction_start_year"] > T, "customer_id"].unique())).astype(int)
first_year = past.groupby("customer_id")["transaction_start_year"].min()
pool["tenure"] = T - pool.index.map(first_year)
pool["보유상품수"] = past.groupby("customer_id").size()
pool["납입액합계"] = np.log1p(past.groupby("customer_id")["transaction_amt"].sum())   # 로그 변환
pool["상품금액합계"] = np.log1p(past.groupby("customer_id")["product_amt"].sum())    # 로그 변환

# ---- 왼쪽: 숫자형 피어슨 ----
num = pool[["label", "tenure", "보유상품수", "납입액합계", "상품금액합계"]].corr()
num.columns = ["라벨", "경과연수", "보유상품수", "납입액(로그)", "상품금액(로그)"]
num.index = num.columns

# ---- 오른쪽: 크라메르 V ----
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n = ct.values.sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

cats = ["라벨", "연령대", "성별", "지역", "소득구간", "직업군", "위험등급", "신용점수대", "결혼"]
cols = ["label", "age", "gender", "region", "income_bracket", "occupation_group",
        "risk_grade", "credit_rating", "marital_status"]
cv = pd.DataFrame(index=cats, columns=cats, dtype=float)
for i in range(len(cols)):
    for j in range(len(cols)):
        cv.iloc[i, j] = cramers_v(pool[cols[i]], pool[cols[j]])

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
for ax, mat, title, vmax in [
    (axes[0], num, "숫자형 피어슨 상관계수 (금액은 로그 변환)", 1.0),
    (axes[1], cv, "범주형-범주형 크라메르 V (연관 강도, 0~1)", 0.8),
]:
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index, fontsize=9)
    ax.set_title(title, fontsize=11)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iloc[i, j]
            ax.text(j, i, f"{v:.2f}" if i != j else "1.0", ha="center", va="center",
                    fontsize=8, color="white" if abs(v) > vmax * 0.55 else "black")
fig.colorbar(im, ax=axes, shrink=0.75)
fig.suptitle("T=2024년 말 풀 38,932명 — 변수 간 상관 히트맵", fontsize=13)
fig.savefig(f"{OUT}\\예측_T2024_상관히트맵.png", dpi=150, bbox_inches="tight")
print("그림 저장: output/예측_T2024_상관히트맵.png")
