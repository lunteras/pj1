# -*- coding: utf-8 -*-
"""3단계 RF 하이퍼파라미터 확정 실험 A안(경량 탐색 -> 확정) — step3_rf_confirm.py

[용도]
    확정 모델인 로지스틱 v3(반복 20회 AUC 0.720±0.010) 대비 랜덤포레스트 튜닝의
    우위가 유지되는지, 그리고 튜닝이 이전 실험의 기본 RF(0.743±0.011)보다 나은지를
    2단계 프로토콜로 확정한다.
      1단계(경량 탐색): 12조합 그리드를 시드 42~46(5회)으로 훑고, 평균 ROC-AUC로
                        최고 조합 1개를 고른다.
      2단계(확정):      최고 조합을 시드 42~61(20회)로 재검증하고, 같은 시드(같은
                        분할)로 로지스틱 v3도 매번 학습해 페어 비교한다.

[프레임] step3_model_v3.py / step3_model_compare.py와 동일
    데이터 : [금융] TL_2.소비자.zip (utf-8-sig), customer_type=='보험' 거래만 사용
    풀     : transaction_start_year<=2024(2024년 말) 상품을 1개 이상 보유한 고객
             38,932명(drop_duplicates('customer_id'), 고객 1행)
    타겟 y : transaction_start_year>2024(2025년) 신규 가입 여부 — 1,137명(2.92%)

[피처 13종] 로지스틱 v3와 동일
    범주 8 원-핫(drop_first): age, gender, region, income_bracket,
        occupation_group, risk_grade, credit_rating, marital_status
    수치 5: tenure(2024-최초가입연도), log1p(거래금액 합), log1p(상품금액 합),
        cnt2(보유상품수>=2), rider('주보험'·'특약' 동시 보유)
    교차 5열: 연령 더미 × 기혼 지시자. 파생값은 모두 2024년 이전 거래만 사용(누수 방지).
    RF는 스케일링 없이 원값 사용, 로지스틱만 v3 전처리(원-핫 + 표준화) 적용.

[그리드 12조합]
    max_depth [6, None] × max_leaf_nodes [16, 32, None] × max_features ['sqrt', 0.5]
    — max_leaf_nodes 세 값은 사용자 지정, max_depth·max_features 요인은 기존 RF
      확정 실험 초안의 프로토콜을 그대로 유지(2×3×2=12조합).
    공통: RandomForestClassifier(n_estimators=300, min_samples_leaf=1,
          class_weight='balanced_subsample', n_jobs=-1, random_state=시드)

[지표] 매 평가 세트에서 ROC-AUC + 상위 500/1000/2000명 실제 가입률(%)

[산출물]
    _rf_log.txt           : 1단계 12조합 표, 최고 조합, 2단계 20회 결과(평균±sd,
                            시드별 상세), 로지스틱 페어 비교, 판정, 중요도 TOP10 (UTF-8)
    output/RF확정_비교.png : 1단계 12조합 AUC 바 + 2단계 AUC 3바(95% 신뢰구간)
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from matplotlib import font_manager
import matplotlib.pyplot as plt

import sklearn
print("scikit-learn", sklearn.__version__, flush=True)

ROOT = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일"
SRC = ROOT + r"\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"
OUT = ROOT + r"\output"
LOG = ROOT + r"\_rf_log.txt"
FIG = OUT + r"\RF확정_비교.png"

names = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

T0 = time.time()

# --- 데이터 준비: step3_model_v3.py와 동일 (T=2024 풀, 피처 13종) ---
fin = pd.read_csv(SRC, encoding="utf-8-sig")
print(f"원본 {fin.shape[0]:,}행 × {fin.shape[1]}열 로드 ({time.time()-T0:.0f}s)", flush=True)
ins = fin[fin["customer_type"] == "보험"].copy()
T = 2024
past = ins[ins["transaction_start_year"] <= T].copy()
fut = ins[ins["transaction_start_year"] > T].copy()

pool = ins.drop_duplicates("customer_id").copy()
pool = pool[pool["customer_id"].isin(past["customer_id"])].copy()
y = pool["customer_id"].isin(set(fut["customer_id"])).astype(int).values

first_year = past.groupby("customer_id")["transaction_start_year"].min()
tenure = (T - pool["customer_id"].map(first_year)).values
cnt = past.groupby("customer_id").size().reindex(pool["customer_id"]).values
cnt2 = (cnt >= 2).astype(float)
log_pay = np.log1p(past.groupby("customer_id")["transaction_amt"].sum().reindex(pool["customer_id"]).values)
log_amt = np.log1p(past.groupby("customer_id")["product_amt"].sum().reindex(pool["customer_id"]).values)
rider = past.groupby("customer_id")["transaction_keyword"].apply(
    lambda s: int(s.astype(str).str.contains("주보험").any() and s.astype(str).str.contains("특약").any())
).reindex(pool["customer_id"]).values

CATS = ["age", "gender", "region", "income_bracket", "occupation_group", "risk_grade", "credit_rating", "marital_status"]
NUMS = ["tenure", "log_pay", "log_amt", "cnt2", "rider"]
numdf = pd.DataFrame({"tenure": tenure, "log_pay": log_pay, "log_amt": log_amt, "cnt2": cnt2, "rider": rider})

age_mar = pd.get_dummies(pool["age"], drop_first=True).astype(float)
age_mar = age_mar.mul(pool["marital_status"].eq("기혼").astype(float), axis=0)
CROSS = list(age_mar.columns)

# RF용 설계행렬: 범주 원-핫(drop_first) + 수치 원값(스케일링 없음) + 교차
dum = pd.get_dummies(pool[CATS].astype(str), drop_first=True).astype(float)
X_rf = pd.concat([dum.reset_index(drop=True), numdf, age_mar.reset_index(drop=True)], axis=1)
FEATS = list(X_rf.columns)

# 로지스틱 기준선용: v3 전처리(범주 원-핫 + 수치 표준화) 그대로
X_lg = pd.concat([pool[CATS].reset_index(drop=True), numdf, age_mar.reset_index(drop=True)], axis=1)
pre_lr = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATS),
    ("num", StandardScaler(), NUMS + CROSS),
])
pipe_lr = Pipeline([("pre", pre_lr),
                    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])

GRID = [(md, mln, mf)
        for md in [6, None]
        for mln in [16, 32, None]
        for mf in ["sqrt", 0.5]]
NAMES = {g: f"depth={g[0]}, nodes={g[1]}, feat={g[2]}" for g in GRID}

S1_SEEDS = list(range(42, 47))   # 1단계: 시드 42~46 (5회)
S2_SEEDS = list(range(42, 62))   # 2단계: 시드 42~61 (20회)
KS = (500, 1000, 2000)


def make_rf(g, seed):
    return RandomForestClassifier(
        n_estimators=300, max_depth=g[0], max_leaf_nodes=g[1], max_features=g[2],
        min_samples_leaf=1, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)


def lift(yv, proba, k):
    o = np.argsort(-proba)[:k]
    return yv[o].mean() * 100


def split(seed):
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
    return (X_rf.iloc[idx_tr], X_rf.iloc[idx_te], X_lg.iloc[idx_tr], X_lg.iloc[idx_te],
            y[idx_tr], y[idx_te])


print(f"풀 {len(y):,}명 | 타겟(2025 가입) {int(y.sum()):,}명({y.mean()*100:.2f}%) | "
      f"평가 세트 {int(0.3*len(y)):,}명 | 피처 열 {X_rf.shape[1]}", flush=True)

# ================= 1단계: 12조합 × 시드 42~46(5회) 경량 탐색 =================
res_s1 = {g: {"auc": [], 500: [], 1000: [], 2000: []} for g in GRID}
t0 = time.time()
for seed in S1_SEEDS:
    Xtr, Xte, _, _, ytr, yte = split(seed)
    for g in GRID:
        rf = make_rf(g, seed)
        rf.fit(Xtr, ytr)
        p = rf.predict_proba(Xte)[:, 1]
        r = res_s1[g]
        r["auc"].append(roc_auc_score(yte, p))
        for k in KS:
            r[k].append(lift(yte, p, k))
    print(f"[1단계] 시드 {seed} 완료 ({time.time()-t0:.0f}s)", flush=True)

mean1 = {g: {m: float(np.mean(res_s1[g][m])) for m in ["auc", 500, 1000, 2000]} for g in GRID}
sd1 = {g: {m: float(np.std(res_s1[g][m])) for m in ["auc", 500, 1000, 2000]} for g in GRID}
best = max(GRID, key=lambda g: mean1[g]["auc"])
print(f"[1단계] 최고 조합: {NAMES[best]} (평균 AUC {mean1[best]['auc']:.4f})", flush=True)

# ============ 2단계: 최고 조합 + 로지스틱 v3 × 시드 42~61(20회) 페어 재검증 ============
res_rf = {"auc": [], 500: [], 1000: [], 2000: [], "imp": []}
res_lr = {"auc": [], 500: [], 1000: [], 2000: []}
per_seed = []
t0 = time.time()
for seed in S2_SEEDS:
    Xtr, Xte, Xl_tr, Xl_te, ytr, yte = split(seed)

    rf = make_rf(best, seed)
    rf.fit(Xtr, ytr)
    p = rf.predict_proba(Xte)[:, 1]
    res_rf["auc"].append(roc_auc_score(yte, p))
    for k in KS:
        res_rf[k].append(lift(yte, p, k))
    res_rf["imp"].append(rf.feature_importances_)

    pipe_lr.fit(Xl_tr, ytr)
    p2 = pipe_lr.predict_proba(Xl_te)[:, 1]
    res_lr["auc"].append(roc_auc_score(yte, p2))
    for k in KS:
        res_lr[k].append(lift(yte, p2, k))

    per_seed.append((seed, res_rf["auc"][-1], res_lr["auc"][-1], res_rf[1000][-1], res_lr[1000][-1]))
    print(f"[2단계] 시드 {seed} 완료 ({time.time()-t0:.0f}s)", flush=True)

m_rf = {m: float(np.mean(res_rf[m])) for m in ["auc", 500, 1000, 2000]}
s_rf = {m: float(np.std(res_rf[m])) for m in ["auc", 500, 1000, 2000]}
m_lr = {m: float(np.mean(res_lr[m])) for m in ["auc", 500, 1000, 2000]}
s_lr = {m: float(np.std(res_lr[m])) for m in ["auc", 500, 1000, 2000]}

d_auc = np.array(res_rf["auc"]) - np.array(res_lr["auc"])
d500 = np.array(res_rf[500]) - np.array(res_lr[500])
d1000 = np.array(res_rf[1000]) - np.array(res_lr[1000])
d2000 = np.array(res_rf[2000]) - np.array(res_lr[2000])


def fmt(m, s, dec=3):
    return f"{m:.{dec}f}±{s:.{dec}f}"


def pair_line(d, dec=4, unit=""):
    lo, hi = d.mean() - 1.96 * d.std(), d.mean() + 1.96 * d.std()
    sig = "유의(CI가 0 제외)" if (lo > 0 or hi < 0) else "유의하지 않음(CI가 0 포함)"
    return (f"차이 {d.mean():+.{dec}f}{unit}±{d.std():.{dec}f}{unit} | "
            f"95% CI [{lo:+.{dec}f}{unit}~{hi:+.{dec}f}{unit}] → {sig} | "
            f"RF 승 {int((d > 0).sum())}/20")


# --- 판정 1: 튜닝 최고 vs 기본 RF 0.743±0.011 (이전 실험, 시드 0~19) ---
REF_M, REF_S = 0.743, 0.011
t_lo, t_hi = m_rf["auc"] - 1.96 * s_rf["auc"], m_rf["auc"] + 1.96 * s_rf["auc"]
r_lo, r_hi = REF_M - 1.96 * REF_S, REF_M + 1.96 * REF_S
if t_lo > r_hi:
    v1 = (f"넘는다 — 튜닝 95% CI 하한 {t_lo:.3f} > 기본 상한 {r_hi:.3f} (구간 분리, "
          f"+{m_rf['auc']-REF_M:.3f})")
elif t_hi < r_lo:
    v1 = f"못 넘는다 — 튜닝 상한 {t_hi:.3f} < 기본 하한 {r_lo:.3f} (구간 분리)"
else:
    v1 = (f"구간이 겹침 — 튜닝 [{t_lo:.3f}~{t_hi:.3f}] vs 기본 [{r_lo:.3f}~{r_hi:.3f}] → "
          f"튜닝 이득을 확정하기 어렵다(차이 없음으로 봄)")

# --- 판정 2: 운용안 갱신 필요 여부 ---
sig1000 = "유의" if (d1000.mean() - 1.96 * d1000.std() > 0 or d1000.mean() + 1.96 * d1000.std() < 0) else "유의하지 않음"
line1000_lo = m_rf[1000] - 1.96 * s_rf[1000]
pass_line = "충족" if line1000_lo >= 5.8 else "미충족"
if sig1000 == "유의" and d1000.mean() > 0 and pass_line == "충족":
    v2 = ("갱신 권고 — 같은 분할 페어에서 RF 튜닝이 로지스틱 대비 상위1000 가입률을 유의하게 "
          "올리고 판정선 5.8%도 충족. 현재 운용안(step4_rank_list, 로지스틱 v3 점수)의 "
          "점수 모델을 RF로 교체하는 갱신을 검토할 만하다.")
elif sig1000 == "유의" and d1000.mean() > 0:
    v2 = ("절충 — RF가 로지스틱보다 유의한 우위이지만 상위1000 판정선 5.8%를 "
          f"충족하지 못한다(CI 하한 {line1000_lo:.2f}%). 갱신은 보류하고 재검토.")
elif d1000.mean() <= 0:
    v2 = "유지 — RF가 로지스틱 대비 우위가 아니므로 운용안(로지스틱 v3) 유지."
else:
    v2 = "유지 — RF의 우위가 유의하지 않으므로 운용안(로지스틱 v3) 유지."

imp_mean = np.mean(res_rf["imp"], axis=0)
top10 = sorted(zip(FEATS, imp_mean), key=lambda t: -t[1])[:10]

# ============================== 결과 로그 (UTF-8) ==============================
L = []
L.append("=== 3단계 RF 하이퍼파라미터 확정 실험 A안(경량 탐색→확정) — step3_rf_confirm.py ===")
L.append(f"실행: {time.strftime('%Y-%m-%d %H:%M')} | 총 소요 {(time.time()-T0)/60:.1f}분 | scikit-learn {sklearn.__version__}")
L.append(f"프레임: 보험 거래만 | 풀 {len(y):,}명(2024년 말 보유 고객) | 타겟 2025 가입 {int(y.sum()):,}명({y.mean()*100:.2f}%)")
L.append(f"평가 세트 {int(0.3*len(y)):,}명(양성 약 {0.3*y.sum():.0f}명) | 피처 열 수 {X_rf.shape[1]}(범주 원-핫 {dum.shape[1]} + 수치 5 + 교차 {len(CROSS)})")
L.append("프로토콜: A안 2단계 — ①12조합×시드42~46(5회) 탐색 → ②최고조합+로지스틱v3×시드42~61(20회) 페어 재검증")
L.append("          stratified 70/30 · RF n_estimators=300, min_samples_leaf=1, class_weight='balanced_subsample'")
L.append("          그리드: max_depth[6,None] × max_leaf_nodes[16,32,None] × max_features[sqrt,0.5]")
L.append("          로지스틱: v3 전처리(원-핫+표준화)+LogisticRegression(max_iter=2000, balanced)")
L.append("")
L.append("--- [1단계] 12조합 탐색 결과 (시드 42~46, 5회 평균±sd) ---")
L.append(f"{'조합':<32} {'ROC-AUC':<15} {'상위500%':<11} {'상위1000%':<11} {'상위2000%':<11}")
for g in sorted(GRID, key=lambda g: -mean1[g]["auc"]):
    r = res_s1[g]
    L.append(f"{NAMES[g]:<32} {fmt(mean1[g]['auc'], sd1[g]['auc']):<15} "
             f"{fmt(np.mean(r[500]), np.std(r[500]), 1):<11} "
             f"{fmt(np.mean(r[1000]), np.std(r[1000]), 1):<11} "
             f"{fmt(np.mean(r[2000]), np.std(r[2000]), 1):<11}")
L.append(f"→ 선정 기준: 1단계 평균 ROC-AUC 최고. 최고 조합: {NAMES[best]} "
         f"(AUC {fmt(mean1[best]['auc'], sd1[best]['auc'])}, 상위1000 {fmt(np.mean(res_s1[best][1000]), np.std(res_s1[best][1000]), 1)}%)")
L.append("")
L.append("--- [2단계] 최고 조합 20회 재검증 vs 로지스틱 v3 (시드 42~61, 페어) ---")
L.append(f"{'모델':<34} {'ROC-AUC':<15} {'상위500%':<11} {'상위1000%':<11} {'상위2000%':<11}")
L.append(f"{'RF 튜닝 최고 [' + NAMES[best] + ']':<34} {fmt(m_rf['auc'], s_rf['auc']):<15} "
         f"{fmt(m_rf[500], s_rf[500], 1):<11} {fmt(m_rf[1000], s_rf[1000], 1):<11} {fmt(m_rf[2000], s_rf[2000], 1):<11}")
L.append(f"{'로지스틱 v3 (페어 기준선)':<34} {fmt(m_lr['auc'], s_lr['auc']):<15} "
         f"{fmt(m_lr[500], s_lr[500], 1):<11} {fmt(m_lr[1000], s_lr[1000], 1):<11} {fmt(m_lr[2000], s_lr[2000], 1):<11}")
L.append("")
L.append("--- 페어 비교 (같은 시드 20쌍, RF − 로지스틱) ---")
L.append(f"ROC-AUC   : {pair_line(d_auc)}")
L.append(f"상위500   : {pair_line(d500, 2, '%p')}")
L.append(f"상위1000  : {pair_line(d1000, 2, '%p')}")
L.append(f"상위2000  : {pair_line(d2000, 2, '%p')}")
L.append("")
L.append("--- 2단계 시드별 상세 ---")
L.append(f"{'seed':>4} {'RF_AUC':>8} {'LR_AUC':>8} {'AUC차이':>9} {'RF_t1000':>9} {'LR_t1000':>9} {'t1000차이':>10}")
for seed, ra, la, r1, l1 in per_seed:
    L.append(f"{seed:>4} {ra:>8.4f} {la:>8.4f} {ra-la:>+9.4f} {r1:>9.2f} {l1:>9.2f} {r1-l1:>+10.2f}")
L.append("")
L.append("--- 참고값 (이전 실험 step3_model_compare.py, 시드 0~19) ---")
L.append("로지스틱 v3: AUC 0.720±0.010 / 상위1000 8.2±0.5%")
L.append("RF기본(min_samples_leaf=20, depth 무제한): AUC 0.743±0.011 / 상위1000 9.8±0.7%")
L.append("")
L.append("--- 판정 ---")
L.append(f"① 튜닝이 기본 RF 0.743을 넘는가: {v1}")
L.append(f"② 로지스틱 대비 상위1000: {d1000.mean():+.2f}%p±{d1000.std():.2f} ({sig1000}) · "
         f"RF 승 {int((d1000>0).sum())}/20 · 판정선 5.8%: RF {pass_line} (CI 하한 {line1000_lo:.2f}%)")
L.append(f"③ 운용안 갱신 필요 여부: {v2}")
L.append("")
L.append("--- 최고 조합 피처 중요도 TOP10 (20회 평균, feature_importances_) ---")
for i, (f, v) in enumerate(top10, 1):
    L.append(f"{i:>2}. {f:<30} {v:.4f}")

report = "\n".join(L)
with open(LOG, "w", encoding="utf-8") as f:
    f.write(report + "\n")
print(report, flush=True)

# ============================== 그림: RF확정_비교.png ==============================
fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6))

# (좌) 1단계 12조합 AUC 바
ax = axes[0]
order = sorted(GRID, key=lambda g: mean1[g]["auc"])
vals1 = [mean1[g]["auc"] for g in order]
labs1 = [NAMES[g] for g in order]
cols1 = ["#B23A2E" if g == best else "#8A8F98" for g in order]
ax.barh(range(12), vals1, color=cols1, height=0.62)
ax.set_yticks(range(12))
ax.set_yticklabels(labs1, fontsize=8.5)
ax.set_xlim(min(vals1) - 0.012, max(vals1) + 0.012)
for i, v in enumerate(vals1):
    ax.annotate(f"{v:.4f}", (v, i), textcoords="offset points", xytext=(4, -3),
                fontsize=8, color="#333")
ax.set_xlabel("ROC-AUC (시드 42~46, 5회 평균)")
ax.set_title("1단계 경량 탐색 — 12조합 × 5회\n(빨강 = 최고 조합, 2단계 진출)")
ax.grid(axis="x", alpha=0.3)

# (우) 2단계 AUC 3바 + 95% CI
ax = axes[1]
vals2 = [m_lr["auc"], m_rf["auc"], REF_M]
errs2 = [1.96 * s_lr["auc"], 1.96 * s_rf["auc"], 1.96 * REF_S]
labs2 = ["로지스틱 v3\n(이번 실험, 20회)", f"RF 튜닝 최고\n{NAMES[best]}", "RF 기본\n(이전 실험 참고, 시드 0~19)"]
cols2 = ["#2F6F8F", "#B23A2E", "#8A8F98"]
ax.bar(range(3), vals2, yerr=errs2, capsize=6, color=cols2, width=0.55,
       error_kw={"lw": 1.3, "ecolor": "#444"})
for i, (v, e) in enumerate(zip(vals2, errs2)):
    ax.annotate(f"{v:.3f}", (i, v + e), textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=10.5, fontweight="bold", color=cols2[i])
ax.set_xticks(range(3))
ax.set_xticklabels(labs2, fontsize=9)
ax.set_ylim(min(v - e for v, e in zip(vals2, errs2)) - 0.015,
            max(v + e for v, e in zip(vals2, errs2)) + 0.02)
ax.set_ylabel("평가 세트 ROC-AUC")
ax.set_title("2단계 확정 — 반복 20회(시드 42~61), 오차막대=95% 신뢰구간\n"
             "로지스틱·RF튜닝은 같은 분할 페어, RF기본은 이전 실험 참고값")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(FIG, dpi=150)
plt.close(fig)
print("그림 저장:", FIG, flush=True)
print(f"완료 — 총 {(time.time()-T0)/60:.1f}분", flush=True)
