# -*- coding: utf-8 -*-
"""모델 비교 실험 — 로지스틱 v3(확정) vs 트리 계열 2개 vs 더미
프레임·타겟·피처·분할은 step3_model_v3.py와 동일하게 재구성
평가: 반복 20회 AUC / 상위 500·1000·2000명 실제 가입률 — 보고서 "모델별 성능 비교" 절의 원본
판정 기준(사전 합의): 상위 1,000명 실제 가입률 >= 5.8%
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier

SRC = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"

fin = pd.read_csv(SRC, encoding="utf-8-sig")
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

age_mar = pd.get_dummies(pool["age"], drop_first=True).astype(float)
age_mar = age_mar.mul(pool["marital_status"].eq("기혼").astype(float), axis=0)
CROSS = list(age_mar.columns)
X = pd.concat([
    pool[CATS].reset_index(drop=True),
    pd.DataFrame({"tenure": tenure, "log_pay": log_pay, "log_amt": log_amt, "cnt2": cnt2, "rider": rider}),
    age_mar.reset_index(drop=True),
], axis=1)

CAT = CATS
NUM = NUMS + CROSS
pre_lr = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                            ("num", StandardScaler(), NUM)])
pre_tree = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                              ("num", "passthrough", NUM)])

MODELS = {
    "더미(전원미가입)": (pre_lr, DummyClassifier(strategy="prior"), None),
    "로지스틱 v3 (확정)": (pre_lr, LogisticRegression(max_iter=2000, class_weight="balanced"), None),
    "랜덤포레스트": (pre_tree, RandomForestClassifier(n_estimators=300, min_samples_leaf=20, class_weight="balanced_subsample", random_state=42, n_jobs=-1), None),
    "LightGBM": (pre_tree, LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=30, class_weight="balanced", random_state=42, verbose=-1), None),
}

N_RUNS = 20
KS = (500, 1000, 2000)
res = {name: {"auc": [], "ap": [], "lifts": {k: [] for k in KS}} for name in MODELS}

for seed in range(N_RUNS):
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
    Xtr, Xte, ytr, yte = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]
    for name, (pre, mdl, _) in MODELS.items():
        pipe = Pipeline([("pre", pre), ("clf", mdl)])
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        res[name]["auc"].append(roc_auc_score(yte, p))
        res[name]["ap"].append(average_precision_score(yte, p))
        o = np.argsort(-p)
        for k in KS:
            res[name]["lifts"][k].append(yte[o[:k]].mean() * 100)

def ci(a):
    a = np.array(a); return f"{a.mean():.3f}±{a.std():.3f} (CI {a.mean()-1.96*a.std():.3f}~{a.mean()+1.96*a.std():.3f})"

print(f"반복 {N_RUNS}회 · 평가 {0.3*len(y):.0f}명(양성 약 {0.3*y.sum():.0f}) · 기준선 {y.mean()*100:.2f}% · 판정선 5.8%")
print(f"{'모델':<18} {'ROC-AUC':<26} {'PR-AUC':<26} {'상위500':<10} {'상위1000':<10} {'상위2000':<10}")
for name in MODELS:
    r = res[name]
    l = {k: f"{np.mean(r['lifts'][k]):.1f}±{np.std(r['lifts'][k]):.1f}" for k in KS}
    print(f"{name:<18} {ci(r['auc']):<26} {ci(r['ap']):<26} {l[500]:<10} {l[1000]:<10} {l[2000]:<10}")

print("\n--- 판정선 5.8% 대비 ---")
for name in MODELS:
    m = np.mean(res[name]["lifts"][1000]); s = np.std(res[name]["lifts"][1000])
    lo = m - 1.96 * s
    print(f"{name:<18} 상위1000 {m:.2f}% (CI하한 {lo:.2f}) → {'충족' if lo >= 5.8 else '미충족'}")

print("\n--- 로지스틱 vs 트리 계열 차이 (상위1000, %p) ---")
base = np.array(res["로지스틱 v3 (확정)"]["lifts"][1000])
for name in ["랜덤포레스트", "LightGBM"]:
    d = np.array(res[name]["lifts"][1000]) - base
    print(f"{name:<18} 평균 {d.mean():+.2f}±{d.std():.2f} — {'유의(신뢰구간 0 제외)' if 0 < d.mean()-1.96*d.std() or d.mean()+1.96*d.std() < 0 else '유의하지 않음'}")
