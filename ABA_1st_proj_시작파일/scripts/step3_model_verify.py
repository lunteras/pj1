# -*- coding: utf-8 -*-
"""모델링 검증 — 리포트 확정 수치 재계산 대조 (T=2024 · 로지스틱 v3 · 20회 프로토콜)
용도: EDA 검증과 동일 방식 — 모델을 처음부터 다시 학습해 리포트 수치와 대조, PASS/FAIL 판정
기대치: 풀 38,932 · 양성 1,137(2.9%) · AUC 0.720±0.010 · t500 9.7±0.8 · t1000 8.2±0.5 · t2000 6.7±0.3
허용오차: AUC ±0.02 / 농도 ±1.2·1.0·0.8%p / 풀·양성은 정확 일치
산출: _model_verify_log.txt (UTF-8) + 종료코드 0(PASS)/1(FAIL)
"""
import io, sys, os, zipfile
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = None
for base, _d, fs in os.walk(ROOT):
    for f in fs:
        if f.startswith("TL_2") and f.endswith(".zip"):
            ZIP = os.path.join(base, f); break
    if ZIP: break
with zipfile.ZipFile(ZIP) as z:
    csv = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    df = pd.read_csv(z.open(csv), encoding="utf-8-sig")
ins = df[df["customer_type"] == "보험"].copy()
pool_ids = ins.loc[ins["transaction_start_year"] <= 2024, "customer_id"].unique()
pool = ins[ins["customer_id"].isin(pool_ids)].drop_duplicates("customer_id").set_index("customer_id").copy()
y = ins[ins["customer_id"].isin(pool_ids)].groupby("customer_id")["transaction_start_year"].apply(lambda s: int((s > 2024).any()))
past = ins[ins["customer_id"].isin(pool_ids) & (ins["transaction_start_year"] <= 2024)]
f = past.groupby("customer_id").agg(tenure=("transaction_start_year", "min"), pay=("transaction_amt", "sum"), pamt=("product_amt", "sum"), pcnt=("product_name", "nunique"))
kw = past.groupby("customer_id")["transaction_keyword"].apply(lambda s: "|".join(s.astype(str)))
f["cnt2"] = (f["pcnt"] >= 2).astype(int)
f["rider"] = (kw.str.contains("특약") & kw.str.contains("주보험")).astype(int)
CATS = ["age", "gender", "region", "income_bracket", "occupation_group", "risk_grade", "credit_rating", "marital_status"]
X = pd.get_dummies(pool[CATS], columns=CATS, drop_first=True)
X["tenure"] = 2024 - f["tenure"]; X["pay"] = np.log1p(f["pay"]); X["pamt"] = np.log1p(f["pamt"])
X["cnt2"] = f["cnt2"]; X["rider"] = f["rider"]
mar = (pool["marital_status"] == "기혼").astype(int)
for c in [c for c in X.columns if c.startswith("age_")]:
    X["cross_" + c] = X[c] * mar
X = X.fillna(0); y = y.reindex(X.index).astype(int)
NUM = ["tenure", "pay", "pamt"]
rows = []
for seed in range(42, 62):
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    sc = StandardScaler().fit(Xt[NUM])
    Xt2 = Xt.copy(); Xv2 = Xv.copy(); Xt2[NUM] = sc.transform(Xt[NUM]); Xv2[NUM] = sc.transform(Xv[NUM])
    m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed).fit(Xt2, yt)
    p = m.predict_proba(Xv2)[:, 1]; order = np.argsort(-p)
    r = {"seed": seed, "auc": roc_auc_score(yv, p)}
    for k in (500, 1000, 2000):
        r[f"t{k}"] = yv.values[order[:k]].mean() * 100
    ids = Xv2.index[Xv2["cnt2"] == 1]
    r["rule"] = yv.loc[ids].mean() * 100 if len(ids) else np.nan; r["rule_n"] = len(ids)
    rows.append(r)
R = pd.DataFrame(rows)
EXP = {"auc": (0.720, 0.02), "t500": (9.7, 1.2), "t1000": (8.2, 1.0), "t2000": (6.7, 0.8)}
fails = []
if len(pool) != 38932: fails.append(f"pool={len(pool)} != 38932")
if int(y.sum()) != 1137: fails.append(f"pos={int(y.sum())} != 1137")
L = []
L.append("== 모델 검증 대조 (반복 20회) ==")
L.append(f"풀 {len(pool)} · 양성 {int(y.sum())} ({y.mean()*100:.2f}%) · 피처 {X.shape[1]}열")
for k, (m0, tol) in EXP.items():
    mean, sd = R[k].mean(), R[k].std()
    ok = abs(mean - m0) <= tol
    L.append(f"{k}: 실측 {mean:.2f}±{sd:.2f}  기대 {m0}  {'OK' if ok else 'FAIL'}")
    if not ok: fails.append(f"{k}={mean:.2f} vs {m0}")
L.append(f"규칙(cnt2) 평균: {R['rule'].mean():.1f}% (n 평균 {R['rule_n'].mean():.0f}) — 참고치 11.6%")
L.append("판정: " + ("PASS" if not fails else "FAIL " + "; ".join(fails)))
log = "\n".join(L)
open("_model_verify_log.txt", "w", encoding="utf-8").write(log)
print(log)
if fails: raise SystemExit(1)