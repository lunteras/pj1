# -*- coding: utf-8 -*-
"""2단계. EDA — 라벨(b: 보장 포함 다중 담보) 기준 속성별 확장 비율 + 규칙 사다리 재계산"""
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
ins = fin[fin["customer_type"] == "보험"]
cust = ins.drop_duplicates("customer_id").copy()

MULTI = ["보장+변액", "보장+기타", "보장+변액+기타"]
pop = cust[cust["cross_coverage"].isin(MULTI + ["보장only"])].copy()
pop["expand"] = pop["cross_coverage"].isin(MULTI).astype(int)
n_pop, n_exp = len(pop), int(pop["expand"].sum())
BASE = n_exp / n_pop * 100
print(f"=== 프레임 ===\n모집단(보장 보유 고객): {n_pop:,} / 확장자: {n_exp:,} / 확장 비율: {BASE:.1f}% / 비확장자: {n_pop-n_exp:,}")

# 병력 3분류 (있음 = 없음·기타 제외한 나머지)
pop["disease3"] = np.where(pop["disease_history"] == "없음", "없음",
                    np.where(pop["disease_history"] == "기타", "기타", "있음"))
# 최초 가입 시기 (고객별 가장 이른 가입 연도)
first_year = ins.groupby("customer_id")["transaction_start_year"].min()
pop["first_year"] = pop["customer_id"].map(first_year)
pop["가입시기"] = pd.cut(pop["first_year"], [0, 2023, 2024, 2025],
                       labels=["2023년 이전", "2024년", "2025년"])
# 고객별 금액 합계 — 행 단위 원본 금액 열은 고객 단위에서 무의미하므로 제거 후 합계로 교체
amt = ins.groupby("customer_id")[["transaction_amt", "product_amt"]].sum()
pop = pop.drop(columns=["transaction_amt", "product_amt"])
pop = pop.merge(amt, left_on="customer_id", right_index=True, how="left")

ATTRS = [
    ("연령대", "age", ["20대", "30대", "40대", "50대", "60대", "70대 이상"]),
    ("소득 구간", "income_bracket", ["3000만원미만", "5000만원미만", "8000만원미만", "10000만원이상"]),
    ("위험등급", "risk_grade", ["A", "B", "C", "D", "E"]),
    ("신용 점수대", "credit_rating", ["900점대이상", "800점대", "700점대", "300점대"]),
    ("성별", "gender", ["여", "남"]),
    ("결혼", "marital_status", ["기혼", "미혼"]),
    ("병력", "disease3", ["없음", "있음"]),
    ("지역", "region", ["경기도", "서울", "경상도", "충청도", "인천", "전라도", "부산", "대전", "대구"]),
    ("직업군", "occupation_group", ["비경제활동인구", "사무종사자", "판매종사자", "전문가 및 관련 종사자",
                              "서비스 종사자", "관리자", "장치·기계조작 및 조립 종사자"]),
]

print("\n=== 속성별 확장 비율 (기타 제외) ===")
gap_rows, cat_tables = [], {}
for name, col, order in ATTRS:
    sub = pop[pop[col].isin(order)]
    excluded = int((~pop[col].isin(order)).sum())
    g = sub.groupby(col)["expand"].agg(["count", "sum", "mean"])
    g.columns = ["고객 수", "확장자", "확장 비율(%)"]
    g["확장 비율(%)"] = (g["확장자"] / g["고객 수"] * 100).round(1)
    g = g.loc[order]
    cat_tables[name] = g
    print(f"\n[{name}] (기타 {excluded:,}명 제외)")
    print(g.to_string())
    hi, lo = g["확장 비율(%)"].idxmax(), g["확장 비율(%)"].idxmin()
    gap = g["확장 비율(%)"].max() - g["확장 비율(%)"].min()
    gap_rows.append((name, hi, g["확장 비율(%)"].max(), lo, g["확장 비율(%)"].min(), round(gap, 1),
                     int(g["고객 수"].min())))
    small = g[g["고객 수"] < 30]
    if len(small):
        print(f"  ※ 30 미만 셀: {list(small.index)}")

gap_df = pd.DataFrame(gap_rows, columns=["속성", "최고", "최고(%)", "최저", "최저(%)", "격차(%p)", "최소 셀 크기"]
                      ).sort_values("격차(%p)", ascending=False)
print("\n=== 가름축 순위 (최고-최저 격차) ===")
print(gap_df.to_string(index=False))

# 가입 시기
print("\n=== 최초 가입 시기별 확장 비율 ===")
g = pop.groupby("가입시기", observed=True)["expand"].agg(["count", "sum", "mean"])
g.columns = ["고객 수", "확장자", "확장 비율(%)"]
g["확장 비율(%)"] = (g["확장자"] / g["고객 수"] * 100).round(1)
print(g.to_string())

# 금액
print("\n=== 금액 요약 (고객별 합계, 확장자 vs 비확장자) ===")
for c, label in [("transaction_amt", "거래 금액 합계(월 납입액 추정)"), ("product_amt", "상품 금액 합계")]:
    m = pop.groupby("expand")[c].median() / 1e4
    a = pop.groupby("expand")[c].mean() / 1e4
    print(f"{label}: 비확장자 중앙값 {m[0]:,.0f}만원 / 확장자 중앙값 {m[1]:,.0f}만원 | 평균 {a[0]:,.0f} vs {a[1]:,.0f}만원")

# 규칙 사다리 (라벨 b 재계산)
print("\n=== 규칙 사다리 (라벨 b, 모집단 66,655) ===")
c1 = pop["age"].isin(["40대", "50대"]) & pop["income_bracket"].isin(["8000만원미만", "10000만원이상"])
c2 = c1 & (pop["risk_grade"] == "A") & (pop["credit_rating"] == "900점대이상")
c3 = c2 & (pop["marital_status"] == "미혼")
c4 = c3 & (pop["gender"] == "여")
ladder = [("0. 없음 (전체)", pd.Series(True, index=pop.index)),
          ("1. + 나이 40~50대 & 소득 상위 2구간", c1),
          ("2. + 위험등급 A & 신용 900점대이상", c2),
          ("3. + 미혼", c3),
          ("4. + 여성", c4)]
lad_rows = []
for name, m in ladder:
    n = int(m.sum()); e = int(pop.loc[m, "expand"].sum())
    lad_rows.append((name, n, e, round(e / n * 100, 1) if n else np.nan))
lad_df = pd.DataFrame(lad_rows, columns=["단계", "후보 수", "확장자", "확장 비율(%)"])
print(lad_df.to_string(index=False))
best_late = lad_df.iloc[2:5]
PANJEONG = best_late["확장 비율(%)"].max()
PANJEONG_STEP = best_late.loc[best_late["확장 비율(%)"].idxmax(), "단계"]
print(f"\n판정선 후보: {PANJEONG_STEP} 의 {PANJEONG}% (후보 {int(best_late.loc[best_late['확장 비율(%)'].idxmax(), '후보 수']):,}명)")

# ---- 그림 1: 주요 4축 확장 비율 2x2 ----
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for ax, (name, *_ ) in zip(axes.flat, ATTRS[:4]):
    g = cat_tables[name]
    ax.bar(g.index.astype(str), g["확장 비율(%)"], color="#4C72B0")
    ax.axhline(BASE, color="crimson", ls="--", lw=1)
    ax.set_title(f"{name}별 확장 비율", fontsize=11)
    ax.set_ylabel("확장 비율(%)")
    ax.tick_params(axis="x", labelsize=9)
    for i, v in enumerate(g["확장 비율(%)"]):
        ax.text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=8)
fig.suptitle("주요 속성별 담보 확장 비율 (빨간 점선 = 전체 15.7%)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}\\보장only_EDA_주요축_확장비율.png", dpi=150)
plt.close(fig)

# ---- 그림 2: 가름축 격차 순위 ----
fig, ax = plt.subplots(figsize=(8.5, 5))
d = gap_df.sort_values("격차(%p)")
ax.barh(d["속성"], d["격차(%p)"], color="#55A868")
for i, (v, hi, hi_p, lo, lo_p) in enumerate(zip(d["격차(%p)"], d["최고"], d["최고(%)"], d["최저"], d["최저(%)"])):
    ax.text(v + 0.15, i, f"{v:.1f}%p ({hi} {hi_p:.1f}% ↔ {lo} {lo_p:.1f}%)", va="center", fontsize=8)
ax.set_xlabel("확장 비율 격차(%p) = 최고 카테고리 - 최저 카테고리")
ax.set_title("속성별 가름축 격차 순위 (기타 응답 제외)")
ax.set_xlim(0, d["격차(%p)"].max() * 1.75)
fig.tight_layout()
fig.savefig(f"{OUT}\\보장only_EDA_가름축_격차순위.png", dpi=150)
plt.close(fig)

# ---- 그림 3: 규칙 사다리 (라벨 b) ----
fig, ax = plt.subplots(figsize=(9.5, 5.5))
x = range(len(lad_df))
ax.bar(x, lad_df["확장 비율(%)"], color="#4C72B0", width=0.55, label="확장 비율(%)")
ax.axhline(PANJEONG, color="crimson", ls="--", lw=1.2, label=f"판정선 {PANJEONG}%")
for i, (r, n) in enumerate(zip(lad_df["확장 비율(%)"], lad_df["후보 수"])):
    ax.text(i, r + 0.3, f"{r:.1f}%", ha="center", fontsize=9)
ax2 = ax.twinx()
ax2.plot(x, lad_df["후보 수"], color="darkorange", marker="o", label="후보 수(오른쪽, 로그)")
ax2.set_yscale("log")
ax2.set_ylabel("후보 수 (로그 눈금)")
for i, n in enumerate(lad_df["후보 수"]):
    ax2.annotate(f"{n:,}", (i, n), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="darkorange")
ax.set_xticks(list(x)); ax.set_xticklabels([s.split(". ")[0] for s in lad_df["단계"]])
ax.set_ylabel("확장 비율(%)")
ax.set_title("규칙 사다리 — 라벨 변경(보장 포함 다중 담보) 후 재계산\n" + "\n".join(lad_df["단계"]), fontsize=10)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}\\보장only_기준선_규칙사다리_재계산.png", dpi=150)
plt.close(fig)

print("\n그림 저장 완료: output/보장only_EDA_주요축_확장비율.png, 보장only_EDA_가름축_격차순위.png, 보장only_기준선_규칙사다리_재계산.png")
