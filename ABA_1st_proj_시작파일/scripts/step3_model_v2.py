# -*- coding: utf-8 -*-
"""3단계 기초 모델링 v2 — 최종 피처(범주8+수치4+교차3) 로지스틱 회귀
판정 기준: 평가 세트 상위 1,000명 중 실제 가입률 >= 5.8% (기준선 2.9%의 2배)
그림: 리프트 곡선(반복 분할 신뢰구간), 계수 상위, 예측 확률 분포"""
import io, sys, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

SRC = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"
OUT = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\output"

from matplotlib import font_manager, pyplot as plt
names = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

# ---------- 데이터 구성 (T=2024) ----------
T = 2024
fin = pd.read_csv(SRC, encoding="utf-8-sig")
ins = fin[fin["customer_type"] == "보험"].copy()
past = ins[ins["transaction_start_year"] <= T]
past_ids = set(past["customer_id"].unique())
fut_ids = set(ins.loc[ins["transaction_start_year"] > T, "customer_id"].unique())
cust = ins.drop_duplicates("customer_id").set_index("customer_id")
pool = cust[cust.index.isin(past_ids)].copy()
pool["label"] = pool.index.isin(fut_ids).astype(int)
prod = past.groupby("customer_id").agg(
    cnt=("product_name", "size"),
    pay_sum=("transaction_amt", "sum"),
    amt_sum=("product_amt", "sum"),
    kws=("transaction_keyword", lambda s: " | ".join(sorted(set(map(str, s))))),
)
pool = pool.join(prod)
first_year = past.groupby("customer_id")["transaction_start_year"].min()
pool["tenure"] = T - pool.index.map(first_year)
pool["cnt2"] = (pool["cnt"] >= 2).astype(int)
pool["log_pay"] = np.log1p(pool["pay_sum"])
pool["log_amt"] = np.log1p(pool["amt_sum"])
pool["rider"] = pool["kws"].apply(lambda s: int(any(("주보험" in k) and ("특약" in k) for k in str(s).split(" | "))))
pool["cnt2_x_logpay"] = pool["cnt2"] * pool["log_pay"]
# 연령×결혼 교차 (연령 원-핫 × 기혼 지시자)
mar = (pool["marital_status"] == "기혼").astype(int)
age_d = pd.get_dummies(pool["age"], prefix="age", drop_first=True).astype(float)
for c in age_d.columns:
    pool[c + "_x_기혼"] = age_d[c].values * mar.values
CROSS_AGE_MAR = [c + "_x_기혼" for c in age_d.columns]

CATS = ["age", "gender", "region", "income_bracket", "occupation_group", "risk_grade", "credit_rating", "marital_status"]
NUMS = ["tenure", "log_pay", "log_amt", "cnt2", "rider", "cnt2_x_logpay"] + CROSS_AGE_MAR
X = pool[CATS + NUMS]
y = pool["label"]
print(f"풀 {len(pool):,} / 양성 {int(y.sum()):,} ({y.mean()*100:.1f}%) / 피처 {X.shape[1]}개 (원-핫 포함 아님)")

def lift_eval(y_true, y_score, ks=(500, 1000, 2000)):
    order = np.argsort(-y_score)
    yt = np.asarray(y_true)[order]
    rows = []
    for k in ks:
        hit = int(yt[:k].sum())
        rows.append((k, hit, hit / k * 100, hit / k / y_true.mean(), hit / y_true.sum() * 100))
    return rows

# ---------- 단일 분할 (seed=42) ----------
X_tr, X_ev, y_tr, y_ev = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATS),
                         ("num", StandardScaler(), NUMS)])
pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
pipe.fit(X_tr, y_tr)
p_ev = pipe.predict_proba(X_ev)[:, 1]
auc = roc_auc_score(y_ev, p_ev); ap = average_precision_score(y_ev, p_ev)
print(f"\n[단일 분할 seed=42] 학습 {len(y_tr):,}(양성 {int(y_tr.sum())}) / 평가 {len(y_ev):,}(양성 {int(y_ev.sum())})")
print(f"ROC-AUC {auc:.3f} / 평균정밀도 {ap:.3f} (무작위 {y_ev.mean():.3f})")
rows = lift_eval(y_ev.values, p_ev)
print("상위 k명 리프트(평가): " + " / ".join(f"{k:,}명 {r:.1f}%({m:.1f}x, 포착{c:.0f}%)" for k, h, r, m, c in rows))
# 규칙 비교 (평가 세트)
X_ev2 = X_ev.copy(); X_ev2["y"] = y_ev.values; X_ev2["p"] = p_ev
rule = X_ev2[X_ev2["cnt2"] == 1]
n_rule = len(rule); r_rule = rule["y"].mean() * 100
same = X_ev2.sort_values("p", ascending=False).head(n_rule)
print(f"[규칙 비교] 보유상품수2+ 규칙: {n_rule:,}명 가입률 {r_rule:.1f}% ({r_rule/(y_ev.mean()*100):.1f}x) / 모델 동규모 상위 {n_rule:,}명: {same['y'].mean()*100:.1f}%")

# ---------- 반복 분할 20회 (신뢰구간) ----------
KS = (500, 1000, 2000)
aucs, lifts = [], {k: [] for k in KS}
for seed in range(20):
    a, b, c, d = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    pp = Pipeline([("pre", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATS),
                                              ("num", StandardScaler(), NUMS)])),
                   ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    pp.fit(a, c)
    s = pp.predict_proba(b)[:, 1]
    aucs.append(roc_auc_score(d, s))
    for k, h, r, m, ca in lift_eval(d.values, s, KS):
        lifts[k].append(r)
aucs = np.array(aucs)
print("\n[반복 분할 20회]")
print(f"  ROC-AUC 평균 {aucs.mean():.3f} ± {aucs.std():.3f} (95% CI {np.percentile(aucs,2.5):.3f}~{np.percentile(aucs,97.5):.3f})")
for k in KS:
    v = np.array(lifts[k])
    print(f"  상위 {k:,}명 가입률: 평균 {v.mean():.1f}% ± {v.std():.1f} (95% CI {np.percentile(v,2.5):.1f}~{np.percentile(v,97.5):.1f})")
verdict = np.percentile(lifts[1000], 2.5) >= 5.8
print(f"  판정 기준(상위 1,000명 >= 5.8%): {'충족' if verdict else '미충족'}")

# ---------- 그림 1: 리프트 곡선 (CI 밴드) ----------
ks_axis = np.arange(100, min(len(y_ev), 3001), 100)
curve = {k: [] for k in ks_axis}
for seed in range(20):
    a, b, c, d = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    pp = Pipeline([("pre", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATS),
                                              ("num", StandardScaler(), NUMS)])),
                   ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    pp.fit(a, c)
    s = pp.predict_proba(b)[:, 1]
    yt = np.asarray(d)[np.argsort(-s)]
    cum = np.cumsum(yt) / np.arange(1, len(yt) + 1) * 100
    for k in ks_axis:
        curve[k].append(cum[k - 1])
lo = [np.percentile(curve[k], 2.5) for k in ks_axis]
hi = [np.percentile(curve[k], 97.5) for k in ks_axis]
mid = [np.mean(curve[k]) for k in ks_axis]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ks_axis, mid, color="#C44E52", lw=2, label="모델 상위 k명 (20회 평균)")
ax.fill_between(ks_axis, lo, hi, color="#C44E52", alpha=0.18, label="95% 신뢰구간")
ax.axhline(2.9, color="gray", ls="--", lw=1.2, label="기준선 2.9% (전체)")
ax.scatter([n_rule], [r_rule], color="#4C72B0", zorder=5, s=45, label=f"규칙(보유수2+) {n_rule:,}명 {r_rule:.1f}%")
ax.axhline(5.8, color="#55A868", ls=":", lw=1.4, label="판정선 5.8%")
for k in KS:
    ax.scatter([k], [np.mean(lifts[k])], color="#C44E52", s=22, zorder=6)
    ax.annotate(f"{np.mean(lifts[k]):.1f}%", (k, np.mean(lifts[k])), xytext=(0, 9), textcoords="offset points", fontsize=9, ha="center", color="#C44E52")
ax.set_xlabel("상담 우선순위 상위 k명"); ax.set_ylabel("실제 가입률(%)")
ax.set_title("3단계 기초 모델 — 상위 k명 실제 가입률 (T=2024, 로지스틱 balanced)\n판정: 상위 1,000명 가입률 95% CI 하한이 5.8% 이상이면 도입 가치", fontsize=11)
ax.legend(fontsize=8); ax.set_xlim(0, 3100)
fig.tight_layout(); fig.savefig(f"{OUT}\\모델_리프트곡선_T2024.png", dpi=150); plt.close(fig)

# ---------- 그림 2: 계수 상위 (방향 참고) ----------
ohe = pipe.named_steps["pre"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(CATS))
feat_names = cat_names + NUMS
coefs = pd.Series(pipe.named_steps["clf"].coef_[0], index=feat_names).sort_values(key=abs, ascending=False)[:15][::-1]
colors = ["#C44E52" if v > 0 else "#4C72B0" for v in coefs.values]
fig, ax = plt.subplots(figsize=(8.4, 5.6))
ax.barh(coefs.index, coefs.values, color=colors)
ax.axvline(0, color="gray", lw=0.8)
ax.set_title("계수 절대값 상위 15개 (표준화 기준 — 방향 참고용, 공선성 3쌍 존재)", fontsize=11)
ax.set_xlabel("계수 (양 = 가입 확률 상승)")
fig.tight_layout(); fig.savefig(f"{OUT}\\모델_계수_T2024.png", dpi=150); plt.close(fig)

# ---------- 그림 3: 예측 확률 분포 ----------
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.hist(p_ev[y_ev == 0], bins=50, alpha=0.6, color="#4C72B0", label="실제 미가입", density=True)
ax.hist(p_ev[y_ev == 1], bins=50, alpha=0.6, color="#C44E52", label="실제 가입", density=True)
ax.set_yscale("log")
ax.set_xlabel("모델 예측 확률"); ax.set_ylabel("밀도 (로그 축)")
ax.set_title(f"예측 확률 분포 — 두 무리의 겹침 정도가 AUC {auc:.3f}의 실체", fontsize=11)
ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}\\모델_확률분포_T2024.png", dpi=150); plt.close(fig)

print("\n그림 저장: output/모델_리프트곡선_T2024.png, 모델_계수_T2024.png, 모델_확률분포_T2024.png")
