# -*- coding: utf-8 -*-
"""티어 경계 민감도 검증 (미실행 후보 3번 — 2026-10-16)
   질문: 운용안 티어 규모(500/1,000/2,000명)는 임의 선택값이다.
         경계 근처 농도가 부드럽게 변하는지(자연 분기), 아니면 급격히 변하는지(경계 리스크) 확인한다.
   방법: v3와 동일한 풀·피처·모델을 20회 분할 재구성(자체검증: AUC 0.720±0.010 재현 확인)하고,
         평가 세트 상위 k명의 실제 가입률을 k 그리드(250~3,000)에서 산출한다.
         경계 민감도 = rate(0.8k) 와 rate(1.2k) 의 격차가 전체 기울기 대비 크지 않은지.
   산출: 표(콘솔) + 그림 output/티어경계_민감도.png + 판정 문구
"""
"""티어 경계 민감도 분석 (2026-10-16).
용도: 운용안의 티어 경계(500·1,000·2,000명)가 자연 분기인지 절벽인지 확인.
프레임: T=2024년 말 풀 38,932명, 타겟=2025년 실제 가입(1,137명, 2.9%).
방법: 로지스틱 v3와 동일 피처(13개)로 랜덤 20회 분할 학습, 평가 세트 상위 k의
실제 가입률(%)을 k=250~3,000 그리드에서 평균±sd로 산출. 경계 민감도는
0.8k vs 1.2k 인접 가입률 격차로 판정. 판정 기준: 인접 격차가 전체 기울기의
2배 이상이면 '경계 주의'. 산출: 콘솔 표 + output/티어경계_민감도.png.
재현: python scripts/step4_tier_sensitivity.py"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

SRC = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"
OUT = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\output"

names = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Malgun Gothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

# ── 풀 재구성 (v3 동일 설계) ─────────────────────────────
fin = pd.read_csv(SRC, encoding="utf-8-sig")
ins = fin[fin["customer_type"] == "보험"]
T = 2024
past = ins[ins["transaction_start_year"] <= T]
fut = ins[ins["transaction_start_year"] > T]
past_ids = set(past["customer_id"])
fut_ids = set(fut["customer_id"])
cust = ins.drop_duplicates("customer_id").set_index("customer_id")
pool = cust[cust.index.isin(past_ids)].copy()
pool["label"] = pool.index.isin(fut_ids).astype(int)
pool["tenure"] = T - pool.index.map(past.groupby("customer_id")["transaction_start_year"].min())
rowcnt = past.groupby("customer_id").size()
pool["cnt2"] = (rowcnt >= 2).astype(int)
pool["log_pay"] = np.log1p(past.groupby("customer_id")["transaction_amt"].sum())
pool["log_amt"] = np.log1p(past.groupby("customer_id")["product_amt"].sum())
kw = past.groupby("customer_id")["transaction_keyword"].agg(lambda s: " ".join(s.fillna("")))
rid = kw.str.contains("주보험") & kw.str.contains("특약")
pool["rider"] = pool.index.map(rid).fillna(False).astype(bool)
CATS = ["age", "gender", "region", "income_bracket", "occupation_group", "risk_grade", "credit_rating", "marital_status"]
age_d = pd.get_dummies(pool["age"], drop_first=True)
mar = (pool["marital_status"] == "기혼").astype(int)
cross_cols = [f"ageXmar_{c}" for c in age_d.columns]
for raw, c in zip(age_d.columns, cross_cols):
    pool[c] = age_d[raw].values * mar.values
NUMS = ["tenure", "log_pay", "log_amt", "cnt2", "rider"] + cross_cols

X = pd.get_dummies(pool[CATS], drop_first=True).astype(float)
X = pd.concat([X, pool[NUMS].astype(float)], axis=1)
y = pool["label"]

KS = [250, 400, 500, 600, 750, 1000, 1250, 1500, 2000, 2500, 3000]
# 경계 판정에 필요한 0.8k/1.2k 지점도 함께 측정 (500→400·600, 1000→800·1200, 2000→1600·2400)
KS_BOUND = sorted({int(b * f) for b in (500, 1000, 2000) for f in (0.8, 1.2)})
KS_ALL = sorted(set(KS) | set(KS_BOUND))
SEEDS = list(range(20))
rates = {k: [] for k in KS_ALL}
aucs = []
for sd in SEEDS:
    tr, ev = train_test_split(pool.index, test_size=0.3, stratify=y, random_state=sd)
    sc = StandardScaler().fit(X.loc[tr])
    Xt = sc.transform(X.loc[tr]); Xe = sc.transform(X.loc[ev])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xt, y.loc[tr])
    p = clf.predict_proba(Xe)[:, 1]
    aucs.append(roc_auc_score(y.loc[ev], p))
    order = pd.Series(p, index=ev).sort_values(ascending=False)
    yy = y.loc[ev]
    for k in KS_ALL:
        kk = min(k, len(ev))
        rates[k].append(yy.loc[order.index[:kk]].mean() * 100)

auc_m, auc_s = np.mean(aucs), np.std(aucs)
print(f"[자체검증] 재구성 모델 AUC = {auc_m:.3f} ± {auc_s:.3f}  (v3 기록 0.720±0.010 — ±0.02 내면 동일 설계로 인정)")
assert abs(auc_m - 0.720) <= 0.02, "재구성 실패 — v3와 다른 모델임. 수치를 신뢰하지 말 것"

print("\n=== 상위 k명 실제 가입률 (반복 20회 평균 ± 표준편차) ===")
rows = []
for k in KS:
    m, s = np.mean(rates[k]), np.std(rates[k])
    rows.append((k, m, s))
    print(f"  {k:>5,}명 : {m:.2f}% ± {s:.2f}   ({m/2.9:.1f}배)")

print("\n=== 경계 민감도 (0.8k vs 1.2k 격차) ===")
# 전체 기울기: k가 250→3000로 갈 때 농도가 내려가는 평균 기울기
slope_all = (rows[0][1] - rows[-1][1]) / (rows[-1][0] - rows[0][0])  # %p per 명
d = dict((k, np.mean(rates[k])) for k in KS_ALL)
verdict = []
for b in (500, 1000, 2000):
    lo, hi = int(b * 0.8), int(b * 1.2)
    gap = d[lo] - d[hi]
    local = gap / (hi - lo)
    ratio = abs(local) / abs(slope_all) if slope_all else 0
    v = "부드러움" if ratio < 1.5 else "경계 주의"
    verdict.append((b, lo, hi, gap, ratio, v))
    print(f"  {b:>5,}명 경계: {lo:,}명 {d[lo]:.2f}% → {hi:,}명 {d[hi]:.2f}%  (격차 {gap:+.2f}%p, 국소기울기 {local:+.4f}%p/명, 전체 대비 {ratio:.1f}배) → {v}")

ok_all = all(x[5] == "부드러움" for x in verdict)
print(f"\n[판정] 경계 민감도: {'전 경계 부드러움 — 티어 규모 자유 조정 가능' if ok_all else '일부 경계 급변 — 운용안에 주의 표기 필요'}")

# ── 그림 ──
fig, ax = plt.subplots(figsize=(8, 5))
ks = [r[0] for r in rows]; m = [r[1] for r in rows]; s = [r[2] for r in rows]
ax.plot(ks, m, "o-", color="#166B5D", lw=2)
ax.fill_between(ks, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)], color="#166B5D", alpha=0.15)
ax.axhline(2.9, ls="--", color="#888", lw=1); ax.text(3050, 3.1, "기준선 2.9%", fontsize=9, ha="right", color="#888")
for b, lo, hi, gap, ratio, v in verdict:
    ax.axvline(b, ls=":", color="#B03636" if v != "부드러움" else "#9AA7BD", lw=1)
    ax.annotate(f"{b:,}", (b, max(m) * 0.99), fontsize=9, ha="center", color="#B03636" if v != "부드러움" else "#9AA7BD")
ax.set_xlabel("상위 k명 (평가 세트)"); ax.set_ylabel("실제 가입률(%)")
ax.set_title("티어 경계 민감도 — 상위 k 농도 곡선 (반복 20회 평균±sd, 세로 점선=티어 경계)")
fig.tight_layout(); fig.savefig(f"{OUT}\\티어경계_민감도.png", dpi=150)
print("그림 저장: output/티어경계_민감도.png")