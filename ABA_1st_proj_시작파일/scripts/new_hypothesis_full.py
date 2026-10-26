# -*- coding: utf-8 -*-
"""새 가설 4개 전체 형식 검증 (A→C→B→D) — T=2024 메인 + T=2023 교차
각 가설: ①숫자(교차표) ②그림 ③다른 이유 확인 ④판정"""
import io, sys, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
warnings.filterwarnings("ignore")
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
ins = fin[fin["customer_type"] == "보험"].copy()
cust = ins.drop_duplicates("customer_id").set_index("customer_id")
pools = {}
for T in [2023, 2024]:
    past = ins[ins["transaction_start_year"] <= T]
    past_ids = set(past["customer_id"].unique())
    fut_ids = set(ins.loc[ins["transaction_start_year"] > T, "customer_id"].unique())
    pool = cust[cust.index.isin(past_ids)].copy()
    pool["label"] = pool.index.isin(fut_ids).astype(int)
    fy = past.groupby("customer_id")["transaction_start_year"].min()
    pool["tenure"] = T - pool.index.map(fy)
    pool["보유상품수"] = past.groupby("customer_id").size()
    pool["납입액합계"] = past.groupby("customer_id")["transaction_amt"].sum()
    pools[T] = pool
    print(f"T={T}: 풀 {len(pool):,} / 양성 {int(pool['label'].sum()):,} / {pool['label'].mean()*100:.1f}%")

def tbl(df, by):
    g = df.groupby(by)["label"].agg(["count", "sum", "mean"])
    g.columns = ["고객 수", "가입자", "가입률(%)"]
    g["가입률(%)"] = (g["가입자"] / g["고객 수"] * 100).round(1)
    return g

print("\n" + "#" * 62)
print("# 가설 A — 반복 구매: 보유상품수2+ vs 1개, 연령 조각 내 교차")
print("#" * 62)
for T in [2024, 2023]:
    p = pools[T]
    p = p.copy(); p["보유"] = np.where(p["보유상품수"] >= 2, "2개 이상", "1개")
    print(f"\n[T={T}] 단순 비교")
    print(tbl(p, "보유").to_string())
    ct = pd.crosstab(p["age"], p["보유"], values=p["label"], aggfunc="mean") * 100
    n = pd.crosstab(p["age"], p["보유"])
    print(f"[T={T}] 연령 조각별 가입률(%) / 셀(고객 수)")
    print(ct.round(1).to_string())
    print("셀 최소:", int(n.values.min()), " / 30미만 셀:")
    small = [f"{i}·{c}={int(n.loc[i, c])}" for i in n.index for c in n.columns if n.loc[i, c] < 30]
    print("   " + (", ".join(small) if small else "없음"))
# ② 그림 A
p = pools[2024].copy(); p["보유"] = np.where(p["보유상품수"] >= 2, "2개 이상", "1개")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
g = tbl(p, "보유")
bars = ax.bar(g.index, g["가입률(%)"], color=["#4C72B0", "#C44E52"], width=0.45)
for b, v in zip(bars, g["가입률(%)"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v}%", ha="center", fontsize=10)
ax.axhline(2.9, color="gray", ls="--", lw=1)
ax.text(1.32, 2.95, "전체 2.9%", fontsize=8, color="gray", ha="right")
ax.set_title("가설 A 단순 비교 (T=2024)", fontsize=11); ax.set_ylabel("가입률(%)")
ax = axes[1]
ct = (pd.crosstab(p["age"], p["보유"], values=p["label"], aggfunc="mean") * 100).round(1)
xs = np.arange(len(ct.index)); w = 0.36
ax.bar(xs - w / 2, ct["1개"], w, label="1개", color="#4C72B0")
ax.bar(xs + w / 2, ct["2개 이상"], w, label="2개 이상", color="#C44E52")
ax.set_xticks(xs); ax.set_xticklabels(ct.index)
ax.set_title("연령 조각 내 비교 — 그림자 여부 확인", fontsize=11)
ax.legend(); ax.set_ylabel("가입률(%)")
fig.suptitle("새 가설 A — 반복 구매 성향이 추가 가입을 이끈다", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}\\새가설_A_보유상품수_연령조각.png", dpi=150); plt.close(fig)

print("\n" + "#" * 62)
print("# 가설 C — 소득 역U의 메커니즘: 납입액 구간으로 맞추면 격차가 줄어드나")
print("#" * 62)
for T in [2024, 2023]:
    p = pools[T].copy()
    p["납입구간"] = pd.qcut(p["납입액합계"], 3, labels=["하", "중", "상"])
    print(f"\n[T={T}] 소득×납입액삼분위 가입률(%)")
    ct = pd.crosstab(p["income_bracket"], p["납입구간"], values=p["label"], aggfunc="mean") * 100
    n = pd.crosstab(p["income_bracket"], p["납입구간"])
    print(ct.reindex(["3000만원미만", "5000만원미만", "8000만원미만", "10000만원이상"]).round(1).to_string())
    print("셀 최소:", int(n.values.min()), " / 30미만 셀 수:", int((n.values < 30).sum()))
p = pools[2024].copy()
p["납입구간"] = pd.qcut(p["납입액합계"], 3, labels=["하", "중", "상"])
fig, ax = plt.subplots(figsize=(7.4, 4.6))
ct = pd.crosstab(p["income_bracket"], p["납입구간"], values=p["label"], aggfunc="mean") * 100
ct = ct.reindex(["3000만원미만", "5000만원미만", "8000만원미만", "10000만원이상"])
for i, col in enumerate(["하", "중", "상"]):
    ax.plot(ct.index, ct[col], marker="o", label=f"납입액 {col}")
ax.set_title("가설 C — 소득 × 납입액 구간별 가입률 (T=2024)\n소득 구간이 다른 무리가 구간 안에서 비슷해지면 '이미 보유'가 원인", fontsize=10)
ax.set_ylabel("가입률(%)"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}\\새가설_C_소득역U_납입액교차.png", dpi=150); plt.close(fig)

print("\n" + "#" * 62)
print("# 가설 B — 결혼×연령 교차: 방향 뒤집힘의 재현 (T=2023 재현 시도)")
print("#" * 62)
for T in [2024, 2023]:
    p = pools[T]
    ct = pd.crosstab(p["age"], p["marital_status"], values=p["label"], aggfunc="mean") * 100
    n = pd.crosstab(p["age"], p["marital_status"])
    print(f"\n[T={T}] 연령×결혼 가입률(%) / 셀 최소 {int(n.values.min())}")
    print(ct.round(1).to_string())
    cells = [f"{i}·{c}={int(n.loc[i, c])}" for i in n.index for c in n.columns if n.loc[i, c] < 30]
    print("30미만 셀: " + (", ".join(cells) or "없음"))

print("\n" + "#" * 62)
print("# 가설 D — 지역 효과가 고객 구성의 그림자인가")
print("#" * 62)
for T in [2024]:
    p = pools[T]
    low = ["대구", "대전", "인천"]
    sub = p[p["region"].isin(low + ["경기도"])]
    print(f"[T={T}] 저지역 3곳 vs 경기도 — 고객 구성 비교")
    for c, lab in [("age", "연령(50대 이상 %)"), ("marital_status", "기혼 %"), ("income_bracket", "소득(5000미만 %)"), ("credit_rating", "신용(300점대 %)")]:
        gg = sub.groupby("region")[c].agg(lambda s: (s == s.value_counts().idxmax()).mean() * 100 if c in ("marital_status",) else np.nan)
    comp = sub.groupby("region").agg(
        중장년=("age", lambda s: s.isin(["50대", "60대", "70대 이상"]).mean() * 100),
        기혼=("marital_status", lambda s: (s == "기혼").mean() * 100),
        소득5천미만=("income_bracket", lambda s: (s == "5000만원미만").mean() * 100),
        가입률=("label", lambda s: s.mean() * 100),
        고객수=("label", "size"),
    ).round(1)
    print(comp.to_string())
    print("\n[T=2024] 연령 조각 안에서 지역별 가입률(%) — 40대·50대 중심 (셀 최소 확인)")
    for ag in ["40대", "50대"]:
        q = p[p["age"] == ag]
        ct2 = pd.crosstab(q["region"], np.where(q["region"].isin(low), "저지역3", "그 외"), values=q["label"], aggfunc="mean") * 100
        n2 = pd.crosstab(q["region"], np.where(q["region"].isin(low), "저지역3", "그 외"))
        print(f"  {ag}: {ct2.round(1).to_dict('index')} 셀 최소 {int(n2.values.min())}")
p = pools[2024]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
comp_full = p.groupby("region").agg(
    가입률=("label", lambda s: s.mean() * 100), 중장년=("age", lambda s: s.isin(["50대", "60대", "70대 이상"]).mean() * 100)).sort_values("가입률")
ax.barh(comp_full.index, comp_full["가입률"], color="#4C72B0")
ax.axvline(2.9, color="gray", ls="--", lw=1)
for i, (r, row) in enumerate(comp_full.iterrows()):
    ax.text(row["가입률"] + 0.05, i, f"{row['가입률']:.1f}%", va="center", fontsize=8)
ax.set_title("지역별 가입률 (T=2024)", fontsize=10); ax.set_xlabel("가입률(%)")
ax = axes[1]
ax.scatter(comp_full["중장년"], comp_full["가입률"], color="#C44E52", s=28)
for r, row in comp_full.iterrows():
    ax.annotate(r, (row["중장년"], row["가입률"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
r = np.corrcoef(comp_full["중장년"], comp_full["가입률"])[0, 1]
ax.set_title(f"구성 그림자 확인 — 중장년 비율 vs 가입률\n상관 r={r:.2f} (높으면 구성 차이가 원인일 가능성)", fontsize=10)
ax.set_xlabel("50대 이상 비율(%)"); ax.set_ylabel("가입률(%)")
fig.suptitle("새 가설 D — 지역 효과는 성향인가, 구성(접점)의 그림자인가", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{OUT}\\새가설_D_지역_구성그림자.png", dpi=150); plt.close(fig)

print("\n그림 저장: output/새가설_A_보유상품수_연령조각.png, 새가설_C_소득역U_납입액교차.png, 새가설_D_지역_구성그림자.png")
