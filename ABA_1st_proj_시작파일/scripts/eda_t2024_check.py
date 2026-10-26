# -*- coding: utf-8 -*-
"""T=2024 프레임 EDA 데이터 점검 4항목 — 결측·이상치·타겟 분포·상관관계"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
import pandas as pd
import numpy as np

SRC = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"
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
pool["납입액합계"] = past.groupby("customer_id")["transaction_amt"].sum()
pool["상품금액합계"] = past.groupby("customer_id")["product_amt"].sum()

# ================== 1. 결측치 비율과 위치 ==================
print("=" * 70)
print("1. 결측치의 비율과 위치 (고객 단위 풀 38,932명)")
print("=" * 70)
miss = pd.DataFrame({
    "완전결측(NaN)": pool.isna().sum(),
    "고유값수": pool.nunique(),
})
miss["완전결측(%)"] = (miss["완전결측(NaN)"] / len(pool) * 100).round(2)

# 대체값(-, 미수집, 미답변, 기타) 위치
SUBST = ["-", "미수집", "미답변"]
rows = []
for c in pool.columns:
    if pool[c].dtype == object:
        s = pool[c].astype(str).str.strip()
        n_subst = int(s.isin(SUBST).sum())
        n_etc = int((s == "기타").sum())
        n_dash = int((s == "-").sum())
        if n_subst + n_etc > 0:
            rows.append((c, n_dash, n_etc, round((n_dash + n_etc) / len(pool) * 100, 1)))
print("\n[대체값 위치] (열 / '-'건수 / '기타'건수 / 합계 비율%)")
for c, d, e, p in rows:
    print(f"  {c:24s}  '-' {d:>6,}  '기타' {e:>6,}  합계 {p:5.1f}%")

print("\n[완전 결측(NaN) 존재 열]")
has_na = miss[miss["완전결측(NaN)"] > 0]
if len(has_na):
    print(has_na[["완전결측(NaN)", "완전결측(%)"]].to_string())
else:
    print("  없음 — 모든 열에 NaN 0건")

print("\n  ※ 참고: 스케줄 히스토리에서 본 미수집 이슈는 transaction_keyword(증권쪽)에 국한 — 보험 고객 pool은 - 기반 재작성이므로 '-'와 '기타' 위치가 위 출력이 정답")

# ================== 2. 이상치와 값의 범위 ==================
print("\n" + "=" * 70)
print("2. 이상치와 값의 범위")
print("=" * 70)
num_cols = ["tenure", "보유상품수", "납입액합계", "상품금액합계"]
for c in num_cols:
    s = pool[c].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    hi = q3 + 1.5 * iqr
    n_out = int((s > hi).sum())
    print(f"\n[{c}] (단위: {'만원' if '금액' in c or '납입' in c else '개수/년'})")
    print(f"  최소 {s.min():,.0f} / Q1 {q1:,.0f} / 중앙값 {s.median():,.0f} / Q3 {q3:,.0f} / 최대 {s.max():,.0f}")
    print(f"  IQR 상한(Q3+1.5IQR) {hi:,.0f} 초과 이상치 {n_out:,}건 ({n_out/len(s)*100:.1f}%)")
    print(f"  상위 0.1%: {s.quantile(0.999):,.0f} / 평균 {s.mean():,.0f} — 평균이 중앙값보다 {'훨씬 큼(우측 꼬리)' if s.mean() > 3*s.median() else '중앙값 근처'}")
print("\n  ※ 납입액합계·상품금액합계는 우측 꼬리(큰 값 몇 개가 평균을 끌어올림) — 모델 입력 시 로그 변환 고려, 3단계에서 결정")

# ================== 3. 대상 변수 분포와 클래스 불균형 ==================
print("\n" + "=" * 70)
print("3. 대상 변수(label) 분포와 클래스 불균형")
print("=" * 70)
vc = pool["label"].value_counts()
print(f"\n  0 (미가입): {vc[0]:,}명 ({vc[0]/len(pool)*100:.1f}%)")
print(f"  1 (2025년 가입): {vc[1]:,}명 ({vc[1]/len(pool)*100:.1f}%)")
print(f"  클래스 비 1 : {vc[0]/vc[1]:.1f} (양성 비율 {vc[1]/len(pool)*100:.1f}%)")
print(f"\n  불균형 처리 옵션: class_weight='balanced' (가중치 비 {vc[0]/vc[1]:.1f}:1 역수 적용)")
print(f"  기준선: 무조건 전원 예측 '미가입'이면 정확도 {vc[0]/len(pool)*100:.1f}% — 정확도만 보면 안 되는 이유")

# ================== 4. 변수 사이의 상관관계 ==================
print("\n" + "=" * 70)
print("4. 변수 사이의 상관관계")
print("=" * 70)
# 숫자형: 라벨 포함
num_df = pool[["label", "tenure", "보유상품수", "납입액합계", "상품금액합계"]].copy()
num_df["로그납입액"] = np.log1p(num_df["납입액합계"])
num_df["로그상품금액"] = np.log1p(num_df["상품금액합계"])
print("\n[숫자형 피어슨 상관계수]")
corr = num_df.corr(numeric_only=True).round(3)
print(corr.to_string())

print("\n[라벨과의 상관 순위]")
lab_corr = corr["label"].drop("label").sort_values(key=abs, ascending=False)
for k, v in lab_corr.items():
    print(f"  {k:12s} {v:+.3f}")

# 범주형-라벨: 상관비(eta squared 대신 간단히 각 범주 평균과 분산 이용) 대신
# 범주형-범주형: 크라메르 V
from scipy.stats import chi2_contingency
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n = ct.values.sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

cats = ["age", "gender", "region", "income_bracket", "occupation_group",
        "risk_grade", "credit_rating", "marital_status"]
print("\n[범주형-범주형 크라메르 V (0.4 이상 쌍만 — 강한 연관)]")
for i in range(len(cats)):
    for j in range(i + 1, len(cats)):
        v = cramers_v(pool[cats[i]], pool[cats[j]])
        if v >= 0.4:
            print(f"  {cats[i]} x {cats[j]}: {v:.3f}")

print("\n[범주형-라벨 연관 (크라메르 V)]")
for c in cats + ["갱신형보유" if "갱신형보유" in pool.columns else "tenure"]:
    try:
        v = cramers_v(pool[c], pool["label"])
        print(f"  {c:20s} {v:.3f}")
    except Exception as e:
        print(f"  {c:20s} 오류 {e}")

print("\n※ 크라메르 V 해서: 0.1 미만 미미, 0.1~0.3 약한 연관, 0.3 이상 강한 연관")
