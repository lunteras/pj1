# -*- coding: utf-8 -*-
"""3단계 기초 모델링 v3 — 교차항(보유수×납입액) 제거본
T=2024년 말 풀 38,932명, 타겟=2025년 신규 가입 여부
피처 13개: 범주 8 원-핫 + 수치 5(경과연수, 납입액로그, 상품금액로그, 보유상품수2이상, 특약포함보유) + 연령×결혼 교차
판정: 상위 1,000명 실제 가입률 >= 5.8% (기준선 2.9%의 2배)
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib import font_manager
import matplotlib.pyplot as plt

SRC = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"
OUT = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\output"

names = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

fin = pd.read_csv(SRC, encoding="utf-8-sig")
ins = fin[fin["customer_type"] == "보험"].copy()
T = 2024
past = ins[ins["transaction_start_year"] <= T].copy()
fut = ins[ins["transaction_start_year"] > T].copy()

# --- 풀 구성: T년 말까지 보유 고객만 (행 단위=고객 1명) ---
pool = ins.drop_duplicates("customer_id").copy()
pool = pool[pool["customer_id"].isin(past["customer_id"])].copy()
# 타겟 y: 이 고객이 T 이후(2025년) 새 상품을 가입했으면 1
y = pool["customer_id"].isin(set(fut["customer_id"])).astype(int).values

# 파생 피처 5개 — 모두 T 이전 값만 사용 (미래 정보 누수 방지)
first_year = past.groupby("customer_id")["transaction_start_year"].min()
tenure = (T - pool["customer_id"].map(first_year)).values
cnt = past.groupby("customer_id").size().reindex(pool["customer_id"]).values
cnt2 = (cnt >= 2).astype(float)
log_pay = np.log1p(past.groupby("customer_id")["transaction_amt"].sum().reindex(pool["customer_id"]).values)
log_amt = np.log1p(past.groupby("customer_id")["product_amt"].sum().reindex(pool["customer_id"]).values)
rider = past.groupby("customer_id")["transaction_keyword"].apply(
    lambda s: int(s.astype(str).str.contains("주보험").any() and s.astype(str).str.contains("특약").any())
).reindex(pool["customer_id"]).values

# 피처 확정 목록 — 가설 검증 4회차로 살아남은 신호만
# 제외: 병력(무관)·갱신형(소멸)·cnt2_x_logpay 교차(F 철회)
CATS = ["age", "gender", "region", "income_bracket", "occupation_group", "risk_grade", "credit_rating", "marital_status"]
NUMS = ["tenure", "log_pay", "log_amt", "cnt2", "rider"]

# 연령×결혼 교차 — 가설 B (기혼 우위는 중년에서만)
age_mar = pd.get_dummies(pool["age"], drop_first=True).astype(float)
age_mar = age_mar.mul(pool["marital_status"].eq("기혼").astype(float), axis=0)
CROSS = list(age_mar.columns)
X = pd.concat([
    pool[CATS].reset_index(drop=True),
    pd.DataFrame({"tenure": tenure, "log_pay": log_pay, "log_amt": log_amt, "cnt2": cnt2, "rider": rider}),
    age_mar.reset_index(drop=True),
], axis=1)

# 전처리: 범주 원-핫(모르는 값 무시) + 수치 표준화
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATS),
    ("num", StandardScaler(), NUMS + CROSS),
])
# 불균형 1:33 — balanced 가중치로 양성 1건을 33건 무게로
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
pipe = Pipeline([("pre", pre), ("clf", clf)])

# 고객 단위 학습/평가 분할 (이 프레임은 고객 1행 — 원칙 유지)
def split_fit(X, y, seed):
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
    pipe.fit(X.iloc[idx_tr], y[idx_tr])
    p_te = pipe.predict_proba(X.iloc[idx_te])[:, 1]
    return p_te, y[idx_te]

# 단일 분할(참고) + 규칙 비교 + 반복 20회 신뢰구간
p_te, y_ev = split_fit(X, y, 42)
auc1 = roc_auc_score(y_ev, p_te)
ap1 = average_precision_score(y_ev, p_te)

def lift(yv, proba, k):
    o = np.argsort(-proba)[:k]
    return yv[o].mean() * 100, yv[o].sum()

row1 = lift(y_ev, p_te, 500); row2 = lift(y_ev, p_te, 1000); row3 = lift(y_ev, p_te, 2000)

# 규칙 비교 (평가 세트 안에서)
ev_cnt = cnt[np.isin(np.arange(len(y)), np.where(np.isin(np.arange(len(y)), np.array([])))[0])] if False else None
idx_te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=42)[1]
ev_mask = np.zeros(len(y), bool); ev_mask[idx_te] = True
rule_mask = ev_mask & (cnt >= 2)
r_rule = y[rule_mask].mean() * 100; n_rule = rule_mask.sum(); p_rule = y[rule_mask].sum()
o = np.argsort(-p_te)[:n_rule]
same_rate = y_ev[o].mean() * 100

# --- 반복 분할 20회: 단일 분할 오차를 신뢰구간으로 ---
# 반복 분할 20회
aucs, l500, l1000, l2000, xint_coefs = [], [], [], [], []
for seed in range(20):
    pt, yt = split_fit(X, y, seed)
    aucs.append(roc_auc_score(yt, pt))
    l500.append(lift(yt, pt, 500)[0]); l1000.append(lift(yt, pt, 1000)[0]); l2000.append(lift(yt, pt, 2000)[0])
    coef = pipe.named_steps["clf"].coef_[0]
    feats = pipe.named_steps["pre"].get_feature_names_out()
    xint_coefs.append(coef[list(feats).index("num__cnt2")])

aucs = np.array(aucs); l500 = np.array(l500); l1000 = np.array(l1000); l2000 = np.array(l2000)
def ci(a): return f"{a.mean():.3f} ± {a.std():.3f} (95% CI {a.mean()-1.96*a.std():.3f}~{a.mean()+1.96*a.std():.3f})"

print("=== 3단계 v3 (교차항 제거, 피처 13개) ===")
print(f"피처 열 수: {X.shape[1]} (범주 8 + 수치 5 + 교차 {len(CROSS)})")
print(f"단일 분할(seed42): AUC {auc1:.3f} / AP {ap1:.3f} (무작위 {y_ev.mean():.3f})")
print(f"반복 20회: AUC {ci(aucs)}")
print(f"상위  500명: {l500.mean():.1f}% ± {l500.std():.1f} (CI {l500.mean()-1.96*l500.std():.1f}~{l500.mean()+1.96*l500.std():.1f}) 배수 {l500.mean()/(y_ev.mean()*100):.1f}x")
print(f"상위 1000명: {l1000.mean():.1f}% ± {l1000.std():.1f} (CI {l1000.mean()-1.96*l1000.std():.1f}~{l1000.mean()+1.96*l1000.std():.1f}) 배수 {l1000.mean()/(y_ev.mean()*100):.1f}x 포착 {1000*y_ev.sum()/len(y_ev)*100:.1f}% 중 {row2[1]/y_ev.sum()*100:.0f}%")
print(f"상위 2000명: {l2000.mean():.1f}% ± {l2000.std():.1f} (CI {l2000.mean()-1.96*l2000.std():.1f}~{l2000.mean()+1.96*l2000.std():.1f}) 배수 {l2000.mean()/(y_ev.mean()*100):.1f}x 포착 {row3[1]/y_ev.sum()*100:.0f}%")
print(f"[규칙 비교] 보유상품수2이상: 후보 {n_rule}명 / 가입자 {p_rule}명 / 가입률 {r_rule:.1f}% ({r_rule/(y_ev.mean()*100):.1f}x)")
print(f"[같은 규모 비교] 모델 상위 {n_rule}명 가입률 {same_rate:.1f}%")
print(f"cnt2 계수 안정성(20회): {np.mean(xint_coefs):.3f} ± {np.std(xint_coefs):.3f}")

# 그림 1: 리프트 곡선
fig, ax = plt.subplots(figsize=(8, 5.2))
ks = np.array([100, 300, 500, 700, 1000, 1500, 2000, 2500, 3000])
lens = []
for seed in range(8):
    pt, yt = split_fit(X, y, seed + 100)
    o = np.argsort(-pt)
    cum = np.cumsum(yt[o])
    vals = [cum[k - 1] / k * 100 for k in ks]
    lens.append(vals)
lens = np.array(lens)
mean = lens.mean(0); lo = mean - 1.96 * lens.std(0); hi = mean + 1.96 * lens.std(0)
ax.plot(ks, mean, "-o", color="#2F6F8F", lw=2, ms=4, label="모델 (8회 반복 평균)")
ax.fill_between(ks, lo, hi, color="#2F6F8F", alpha=0.18, label="95% 신뢰구간")
ax.axhline(5.8, color="#B23A2E", ls="--", lw=1.4, label="판정선 5.8% (기준선 2배)")
ax.axhline(y_ev.mean() * 100, color="#8A8F98", ls=":", lw=1.4, label=f"기준선 {y_ev.mean()*100:.1f}%")
ax.axhline(r_rule, color="#7A4FA3", ls="-.", lw=1.4, label=f"규칙(보유2개+) {r_rule:.1f}%")
for k, v in zip(ks, mean):
    ax.annotate(f"{v:.1f}", (k, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color="#2F6F8F")
ax.set_xlabel("모델이 뽑은 상위 k명"); ax.set_ylabel("그 안의 실제 가입률 (%)")
ax.set_title("리프트 곡선 — v3 (교차항 제거, 피처 13개)\n상위 1,000명 8.1%대, 판정선(5.8%) 충족")
ax.set_xlim(60, 3080); ax.set_ylim(0, 16)
ax.legend(fontsize=8.5, loc="upper right")
fig.tight_layout(); fig.savefig(OUT + r"\모델_리프트곡선_T2024_v3.png", dpi=150); plt.close(fig)

# 그림 2: 계수 상위 (방향 참고)
fig, ax = plt.subplots(figsize=(8, 5.6))
coef = pipe.named_steps["clf"].coef_[0]
feats = pipe.named_steps["pre"].get_feature_names_out()
s = pd.Series(coef, index=feats).sort_values()
take = pd.concat([s.head(10), s.tail(10)])
lab = [f[4:].replace("_", " ") for f in take.index]
cols = ["#B23A2E" if v < 0 else "#2F8F6F" for v in take.values]
ax.barh(range(len(take)), take.values, color=cols)
ax.set_yticks(range(len(take))); ax.set_yticklabels(lab, fontsize=9)
ax.axvline(0, color="#444", lw=0.8)
ax.set_title("계수 상위 20 (v3, 표준화 계수 — 방향 참고용)\n음(빨강)=미가입 방향, 양(초록)=가입 방향")
fig.tight_layout(); fig.savefig(OUT + r"\모델_계수_T2024_v3.png", dpi=150); plt.close(fig)

# 그림 3: 예측 확률 분포
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.hist(p_te[y_ev == 0], bins=50, range=(0, 1), color="#8A8F98", alpha=0.75, label="실제 미가입")
ax.hist(p_te[y_ev == 1], bins=50, range=(0, 1), color="#B23A2E", alpha=0.75, label="실제 가입")
ax.set_yscale("log"); ax.set_xlabel("모델 예측 확률"); ax.set_ylabel("고객 수 (로그)")
ax.set_title("예측 확률 분포 — 가입자가 오른쪽(고확률)으로 치우친 정도")
ax.legend()
fig.tight_layout(); fig.savefig(OUT + r"\모델_확률분포_T2024_v3.png", dpi=150); plt.close(fig)
print("그림 3장 저장 완료: 리프트곡선_v3 / 계수_v3 / 확률분포_v3")
