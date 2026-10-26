# -*- coding: utf-8 -*-
"""다음 절차 3개 — 계산·검증 후 리포트 6·8절 반영 자료 산출
B-5: 최근 가입 상품 → 다음 가입 상품 패턴 (자기상관)
B-3: 모델 확률 보정 곡선 (calibration)
B-4: 월 단위 가입 시즌성
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from matplotlib import font_manager
import matplotlib.pyplot as plt

SRC = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"
OUT = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\output"

names = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

fin = pd.read_csv(SRC, encoding="utf-8-sig")
ins = fin[fin["customer_type"] == "보험"].copy()

print("=" * 60)
# --- B-5 자기상관: 최근 가입 상품 → 다음 가입 상품 패턴 ---
print("B-5. 최근 가입 상품 → 다음 가입 상품 패턴")
print("=" * 60)
# 최근(2025) 가입 상품군 분포: 신규 가입자가 무엇을 샀는가
fut = ins[ins["transaction_start_year"] == 2025]
print("\n[2025년 신규 가입 상품 TOP15]")
g = fut.groupby("product_name").size().sort_values(ascending=False).head(15)
for name, n in g.items():
    print(f"  {name}: {n}건")
# keyword 구성
kw = fut["transaction_keyword"].value_counts().head(8)
print("\n[2025년 신규 가입 상품 구성(주보험/특약)]")
for k, n in kw.items():
    print(f"  {k}: {n}")

# T 시점 상품 계열 → 2025 신규 상품 계열 전이 (계열 단순화: 건강/암/실손/연금/저축 등 토큰 기반)
def fam(name):
    s = str(name)
    if "케어" in s: return "M-케어(건강·종합)"
    if "암" in s: return "암보험"
    if "연금" in s or "연금저축" in s: return "연금·저축"
    if "변액" in s or "유니버설" in s: return "변액·유니버설"
    if "실손" in s: return "실손"
    if "상해" in s or "교통" in s or "운전" in s: return "상해·교통"
    if "어린이" in s or "아이" in s or "태아" in s: return "어린이·태아"
    return "기타"

past = ins[ins["transaction_start_year"] <= 2024]
joined = past.merge(fut[["customer_id", "product_name"]], on="customer_id", suffixes=("_t", "_new"))
joined["fam_t"] = joined["product_name_t"].map(fam)
joined["fam_n"] = joined["product_name_new"].map(fam)
print("\n[T 시점 보유 계열 → 2025 신규 계열 전이 (행 기반, 상위 10)]")
tr = pd.crosstab(joined["fam_t"], joined["fam_n"])
tot = joined.groupby("fam_t").size()
for ft in tr.index:
    if tot[ft] < 100: continue
    top = tr.loc[ft].sort_values(ascending=False)
    s = ", ".join(f"{fn} {v}({v/tot[ft]*100:.0f}%)" for fn, v in top.head(4).items() if v > 0)
    print(f"  {ft} ({tot[ft]}행): {s}")

print()
print("=" * 60)
print("B-3. 모델 확률 보정 곡선 (calibration)")
print("=" * 60)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

T = 2024
past24 = ins[ins["transaction_start_year"] <= T].copy()
fut25 = ins[ins["transaction_start_year"] > T].copy()
pool = ins.drop_duplicates("customer_id").copy()
pool = pool[pool["customer_id"].isin(past24["customer_id"])].copy()
y = pool["customer_id"].isin(set(fut25["customer_id"])).astype(int).values
first_year = past24.groupby("customer_id")["transaction_start_year"].min()
tenure = (T - pool["customer_id"].map(first_year)).values
cnt = past24.groupby("customer_id").size().reindex(pool["customer_id"]).values
cnt2 = (cnt >= 2).astype(float)
log_pay = np.log1p(past24.groupby("customer_id")["transaction_amt"].sum().reindex(pool["customer_id"]).values)
log_amt = np.log1p(past24.groupby("customer_id")["product_amt"].sum().reindex(pool["customer_id"]).values)
rider = past24.groupby("customer_id")["transaction_keyword"].apply(
    lambda s: int(s.astype(str).str.contains("주보험").any() and s.astype(str).str.contains("특약").any())
).reindex(pool["customer_id"]).values
CATS = ["age", "gender", "region", "income_bracket", "occupation_group", "risk_grade", "credit_rating", "marital_status"]
NUMS = ["tenure", "log_pay", "log_amt", "cnt2", "rider"]
age_mar = pd.get_dummies(pool["age"], drop_first=True).astype(float).mul(pool["marital_status"].eq("기혼").astype(float), axis=0)
X = pd.concat([pool[CATS].reset_index(drop=True),
               pd.DataFrame({"tenure": tenure, "log_pay": log_pay, "log_amt": log_amt, "cnt2": cnt2, "rider": rider}),
               age_mar.reset_index(drop=True)], axis=1)

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATS),
    ("num", StandardScaler(), NUMS + list(age_mar.columns)),
])
pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])

# 반복 20회 — 평가 세트에서 확률 모아 보정 곡선 + 티어별 실빈도
all_bins, all_probs, all_ys = [], [], []
tier_hits = {k: [] for k in [500, 1000, 2000]}
n_ev = int(len(y) * 0.3)
for seed in range(20):
    it, ie = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
    pipe.fit(X.iloc[it], y[it])
    pr = pipe.predict_proba(X.iloc[ie])[:, 1]
    yr = y[ie]
    all_probs.append(pr); all_ys.append(yr)
    pb, pp = calibration_curve(yr, pr, n_bins=5, strategy="quantile")
    all_bins.append((pb, pp))
    o = np.argsort(-pr)
    for k in tier_hits: tier_hits[k].append(yr[o[:k]].mean() * 100)

# 확률 5분위별 실제 가입률 (20회 평균)
probmat = np.vstack(all_probs); ymat = np.vstack(all_ys)
qedges = np.quantile(probmat.mean(0), [0, .2, .4, .6, .8, 1.0])
qq = np.clip(np.searchsorted(qedges, probmat.mean(0), side="right") - 1, 0, 4)
print("\n[모델 확률 5분위 → 실제 가입률 (20회 평균)]")
print("  분위 | 예측확률 중앙값 | 실제 가입률 | 고객수(평가세트기준)")
for q in range(5):
    m = qq == q
    pr_mid = np.median(probmat.mean(0)[m]) * 100
    rate = ymat.mean(0)[m].mean() * 100
    print(f"  Q{q+1} | {pr_mid:5.2f}% | {rate:5.2f}% | {m.sum()}")

print("\n[티어별 실제 가입률 — 보정 확인용 (20회 평균)]")
for k, v in tier_hits.items():
    print(f"  상위 {k:>5}명: {np.mean(v):5.2f}% ± {np.std(v):4.2f}")

# 보정 그림
fig, ax = plt.subplots(figsize=(6.5, 5))
mids = [0.5, 1.5, 2.5, 3.5, 4.5]
preds = [np.median(probmat.mean(0)[qq == q]) * 100 for q in range(5)]
acts = [ymat.mean(0)[qq == q].mean() * 100 for q in range(5)]
ax.plot([0, 12], [0, 12], "--", color="#888", lw=1, label="완벽 보정 (y=x)")
ax.plot(preds, acts, "o-", color="#2456a6", lw=2, ms=8, label="모델 (5분위)")
ax.set_xlabel("모델이 매긴 확률 (%) — '이 고객과 비슷한 고객의 과거 가입 비율'")
ax.set_ylabel("실제 가입률 (%)")
ax.set_title("보정 곡선 — 모델 확률 vs 실제 빈도 (20회 평균)")
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT + r"\다음절차_B3_보정곡선.png", dpi=150)
plt.close(fig)
print("\n그림 저장: output/다음절차_B3_보정곡선.png")

print()
print("=" * 60)
print("B-4. 월 단위 가입 시즌성")
print("=" * 60)
# 2025년 가입자의 월 분포 — 데이터에 '월'이 있는지 확인
has_month = "transaction_start_month" in ins.columns
print("월 열 존재:", has_month, "| 열 목록:", [c for c in ins.columns if "month" in c.lower() or "date" in c.lower()])
# 연도별 가입 건수로 시즌성 근사 (연도 단위 밖에 없으면 한계 명시)
if not has_month:
    print("\n※ 데이터에 월 단위가 없음 — 연도별 분포로 근사만 가능")
    yc = ins["transaction_start_year"].value_counts().sort_index()
    print("[연도별 가입 건수]")
    for yy, n in yc.items():
        print(f"  {yy}: {n}건")
