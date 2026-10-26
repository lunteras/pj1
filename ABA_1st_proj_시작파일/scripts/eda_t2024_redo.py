# -*- coding: utf-8 -*-
"""T=2024 프레임 EDA 재실행 (정정·보강판) — 결측/이상치/타겟분포/상관 + 그림"""
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

# T=2024 프레임 재계산 — 리포트 5절 수치의 원본 (실측 기준)
fin = pd.read_csv(SRC, encoding="utf-8-sig")
ins = fin[fin["customer_type"] == "보험"].copy()
T = 2024
past = ins[ins["transaction_start_year"] <= T].copy()
cust = ins.drop_duplicates("customer_id").set_index("customer_id")
past_ids = set(past["customer_id"].unique())
fut_ids = set(ins.loc[ins["transaction_start_year"] > T, "customer_id"].unique())
pool = cust[cust.index.isin(past_ids)].copy()
pool["label"] = pool.index.isin(fut_ids).astype(int)
first_year = past.groupby("customer_id")["transaction_start_year"].min()
pool["tenure"] = T - pool.index.map(first_year)
pool["보유상품수"] = past.groupby("customer_id").size()
pool["납입액합계"] = past.groupby("customer_id")["transaction_amt"].sum()   # 원 단위
pool["상품금액합계"] = past.groupby("customer_id")["product_amt"].sum()    # 원 단위
N = len(pool)

print("=" * 66)
print("1. 결측치의 비율과 위치 (고객 단위 풀 {:,}명)".format(N))
print("=" * 66)
print(f"완전 결측(NaN) 있는 열: {(pool.isna().sum() > 0).sum()}개 / NaN 총 {int(pool.isna().sum().sum())}건")
print("\n[대체값 위치]")
for c in [col for col in pool.columns if pool[col].dtype == object]:
    s = pool[c].astype(str).str.strip()
    d = int((s == "-").sum()); e = int((s == "기타").sum())
    if d + e > 0:
        print(f"  {c:22s} '-' {d:>6,} ({d/N*100:5.1f}%)  '기타' {e:>6,} ({e/N*100:5.1f}%)")

print("\n" + "=" * 66)
print("2. 이상치와 값의 범위 (금액: 원 단위, 괄호 = 만원 환산)")
print("=" * 66)
for c, unit_note in [("tenure", "년"), ("보유상품수", "개"), ("납입액합계", "금액"), ("상품금액합계", "금액")]:
    s = pool[c].dropna()
    q1, q3 = s.quantile(.25), s.quantile(.75)
    hi = q3 + 1.5 * (q3 - q1)
    n_out = int((s > hi).sum())
    print(f"\n[{c}] ({'연' if unit_note=='년' else '개' if unit_note=='개' else '원'})")
    if unit_note == "금액":
        print(f"  최소 {s.min():>12,.0f} ({s.min()/1e4:>7,.0f}만원)  Q1 {q1:>12,.0f} ({q1/1e4:>7,.0f}만원)  중앙값 {s.median():>12,.0f} ({s.median()/1e4:>7,.0f}만원)")
        print(f"  Q3 {q3:>12,.0f} ({q3/1e4:>7,.0f}만원)  최대 {s.max():>12,.0f} ({s.max()/1e4:>7,.0f}만원)  평균 {s.mean():>12,.0f} ({s.mean()/1e4:>7,.0f}만원)")
    else:
        print(f"  최소 {s.min():,.0f}  Q1 {q1:,.0f}  중앙값 {s.median():,.0f}  Q3 {q3:,.0f}  최대 {s.max():,.0f}  평균 {s.mean():.2f}")
    print(f"  IQR 상한 {hi:,.0f} 초과 {n_out:,}건 ({n_out/N*100:.1f}%)")

print("\n" + "=" * 66)
print("3. 대상 변수(label)의 분포와 클래스 불균형")
print("=" * 66)
vc = pool["label"].value_counts()
print(f"  0 (2025년 미가입): {vc[0]:>7,}명 ({vc[0]/N*100:.1f}%)")
print(f"  1 (2025년 가입)  : {vc[1]:>7,}명 ({vc[1]/N*100:.1f}%)")
print(f"  클래스 비 1 : {vc[0]/vc[1]:.1f}   무조건 '미가입' 정확도 {vc[0]/N*100:.1f}% → 정확도 평가 무의미")

print("\n" + "=" * 66)
print("4. 변수 사이의 상관관계")
print("=" * 66)
num_df = pd.DataFrame({"라벨": pool["label"], "경과연수": pool["tenure"], "보유상품수": pool["보유상품수"],
                       "납입액(로그)": np.log1p(pool["납입액합계"]), "상품금액(로그)": np.log1p(pool["상품금액합계"])})
corr = num_df.corr().round(3)
print("\n[숫자형 피어슨 (금액 로그 변환)]")
print(corr.to_string())
lab = corr["라벨"].drop("라벨").sort_values(key=abs, ascending=False)
print("\n[라벨과의 상관 순위] " + " / ".join(f"{k} {v:+.3f}" for k, v in lab.items()))

def cramers_v(x, y):
    ct = pd.crosstab(x, y); chi2 = chi2_contingency(ct)[0]; n = ct.values.sum(); r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

cats = ["age", "gender", "region", "income_bracket", "occupation_group", "risk_grade", "credit_rating", "marital_status"]
strong = []
for i in range(len(cats)):
    for j in range(i + 1, len(cats)):
        v = cramers_v(pool[cats[i]], pool[cats[j]])
        if v >= 0.4: strong.append((cats[i], cats[j], v))
print("\n[범주형 강연관 (크라메르 V 0.4 이상)] " + " / ".join(f"{a}×{b} {v:.3f}" for a, b, v in strong))
labv = {c: cramers_v(pool[c], pool["label"]) for c in cats}
print("[범주형-라벨 연관] " + " / ".join(f"{c} {v:.3f}" for c, v in sorted(labv.items(), key=lambda kv: -kv[1])))

print("\n" + "=" * 66)
print("5. 범주형별 2025년 가입률 (모델 피처 참조용 통합)")
print("=" * 66)
ORDER = {"age": ["20대", "30대", "40대", "50대", "60대", "70대 이상"],
         "income_bracket": ["3000만원미만", "5000만원미만", "8000만원미만", "10000만원이상"],
         "risk_grade": ["A", "B", "C", "D", "E"],
         "credit_rating": ["900점대이상", "800점대", "700점대", "300점대"],
         "gender": ["여", "남"], "marital_status": ["기혼", "미혼"]}
for c in cats:
    g = pool.groupby(c)["label"].agg(["count", "sum", "mean"])
    g.columns = ["고객 수", "가입자", "가입률(%)"]
    g["가입률(%)"] = (g["가입자"] / g["고객 수"] * 100).round(1)
    if c in ORDER:
        g = g.reindex([x for x in ORDER[c] if x in g.index])
        rest = g[~g.index.isin(ORDER[c])]
        g = pd.concat([g, rest]) if len(rest) else g
    else:
        g = g.sort_values("가입률(%)", ascending=False)
    items = " / ".join(f"{idx} {int(r['고객 수']):,}명 {r['가입률(%)']}%" for idx, r in g.iterrows())
    print(f"  [{c}] {items}")

# ---------- 그림 A: 타겟 분포 ----------
fig, ax = plt.subplots(figsize=(6, 4.2))
bars = ax.bar(["0 미가입", "1 2025년 가입"], [vc[0], vc[1]], color=["#9BB7D4", "#C44E52"], width=0.5)
for b, v in zip(bars, [vc[0], vc[1]]):
    ax.text(b.get_x() + b.get_width() / 2, v + N * 0.008, f"{v:,}명 ({v/N*100:.1f}%)", ha="center", fontsize=10)
ax.set_yscale("log"); ax.set_ylabel("고객 수 (로그 눈금)")
ax.set_title(f"타겟 분포 — 클래스 불균형 1 : {vc[0]/vc[1]:.1f}\n(풀 {N:,}명 = 2024년 말 보유 기존 고객)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}\\EDA_T2024_타겟분포.png", dpi=150); plt.close(fig)

# ---------- 그림 B: 수치형 4패널 ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
ax = axes[0, 0]
ax.hist(pool["납입액합계"], bins=50, color="#4C72B0")
ax.set_xscale("log"); ax.axvline(pool["납입액합계"].median(), color="crimson", ls="--", lw=1.2)
ax.set_title(f"납입액 합계 (로그 축) — 중앙값 {pool['납입액합계'].median()/1e4:,.0f}만원", fontsize=10)
ax.set_xlabel("원"); ax.set_ylabel("고객 수")
ax = axes[0, 1]
ax.hist(pool["상품금액합계"], bins=50, color="#4C72B0")
ax.set_xscale("log"); ax.axvline(pool["상품금액합계"].median(), color="crimson", ls="--", lw=1.2)
ax.set_title(f"상품 금액 합계 (로그 축) — 중앙값 {pool['상품금액합계'].median()/1e4:,.0f}만원", fontsize=10)
ax.set_xlabel("원"); ax.set_ylabel("고객 수")
ax = axes[1, 0]
tt = pool["tenure"].clip(upper=10)
vc_t = tt.value_counts().sort_index()
ax.bar([str(int(i)) for i in vc_t.index], vc_t.values, color="#55A868")
ax.set_title("가입 후 경과 연수 (10 = 10년 이상)", fontsize=10)
ax.set_xlabel("연"); ax.set_ylabel("고객 수")
for i, v in enumerate(vc_t.values):
    ax.text(i, v + N * 0.003, f"{v:,.0f}", ha="center", fontsize=7)
ax = axes[1, 1]
vc_p = pool["보유상품수"].value_counts().sort_index()
ax.bar([str(int(i)) for i in vc_p.index], vc_p.values, color="#55A868")
ax.set_title("보유 상품 수 — 2개 이상이 라벨 최고 신호", fontsize=10)
ax.set_xlabel("개"); ax.set_ylabel("고객 수")
for i, v in enumerate(vc_p.values):
    ax.text(i, v + N * 0.003, f"{v:,.0f}", ha="center", fontsize=8)
fig.suptitle("수치형 변수 분포 — T=2024년 말 풀 (금액은 우측 꼬리라 로그 축)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{OUT}\\EDA_T2024_수치형분포.png", dpi=150); plt.close(fig)

print("\n그림 저장: output/EDA_T2024_타겟분포.png, output/EDA_T2024_수치형분포.png (+ 기존 예측_T2024_상관히트맵.png)")
