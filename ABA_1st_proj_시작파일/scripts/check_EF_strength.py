# -*- coding: utf-8 -*-
"""가설 E(특약)·F(보유수x납입액 교차)의 논리 기반 강도 점검
1) 소거 실험(모델에서 피처를 빼면 리프트가 얼마나 내려가나) — 피처의 실제 기여
2) E 용량반응(특약 개수) + 같은 상품 계열(M-케어) 안에서의 효과 유지 여부
3) F 교차의 구성 그림자: 2개 보유·저납입 셀의 상품 유형, 보장형 한정 재확인(상/하반)
4) E와 F의 얽힘(상관) 확인"""
import io, sys, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from matplotlib import font_manager, pyplot as plt
names = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2")
SRC = next(ROOT.rglob("TL_2.소비자.zip"))
OUT = ROOT / "ABA_1st_proj_시작파일" / "output"
T = 2024
K = 1000  # 상위 k

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
pool["r_cnt"] = pool["kws"].apply(lambda s: sum(int(("주보험" in k) and ("특약" in k)) for k in str(s).split(" | ")))
pool["cnt2_x_logpay"] = pool["cnt2"] * pool["log_pay"]
mar = (pool["marital_status"] == "미혼").astype(int)
age_d = pd.get_dummies(pool["age"], prefix="age", drop_first=True).astype(float)
for c in age_d.columns:
    pool[c + "_x_미혼"] = age_d[c].values * mar.values
CROSS_AGE_MAR = [c + "_x_미혼" for c in age_d.columns]

CATS = ["age", "gender", "region", "income_bracket", "occupation_group", "risk_grade", "credit_rating", "marital_status"]
NUMS = ["tenure", "log_pay", "log_amt", "cnt2", "rider", "cnt2_x_logpay"] + CROSS_AGE_MAR
y = pool["label"].values

def make_pipe(num_cols):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATS),
        ("num", StandardScaler(), num_cols),
    ])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])

CONFIGS = {
    "전체(기준)": NUMS,
    "-특약(E)": [c for c in NUMS if c != "rider"],
    "-교차항(F)": [c for c in NUMS if c != "cnt2_x_logpay"],
    "-둘다(E+F)": [c for c in NUMS if c not in ("rider", "cnt2_x_logpay")],
    "-cnt2(A)": [c for c in NUMS if c not in ("cnt2", "cnt2_x_logpay")],
}
NREP = 20
res = {k: {"auc": [], "lift": [], "xint": [], "rider_c": []} for k in CONFIGS}
rng_splits = [(42 + i, 42 + i) for i in range(NREP)]
for tr_idx, se in rng_splits:
    itr, ite = train_test_split(np.arange(len(pool)), test_size=0.3, random_state=se)
    for name, cols in CONFIGS.items():
        pipe = make_pipe(cols)
        pipe.fit(pool.iloc[itr][CATS + cols], y[itr])
        pr = pipe.predict_proba(pool.iloc[ite][CATS + cols])[:, 1]
        yt = y[ite]
        res[name]["auc"].append(roc_auc_score(yt, pr))
        top = np.argsort(-pr)[:K]
        res[name]["lift"].append(100.0 * yt[top].mean())
        lr = pipe.named_steps["lr"]
        fnames = list(pipe.named_steps["pre"].get_feature_names_out())
        co = dict(zip(fnames, lr.coef_[0]))
        if "num__cnt2_x_logpay" in co:
            res[name]["xint"].append(co["num__cnt2_x_logpay"])
        if "num__rider" in co:
            res[name]["rider_c"].append(co["num__rider"])

print("=== 1) 소거 실험 (상위 %d명 실제 가입률 %% — 반복 %d회, 평균±표준편차) ===" % (K, NREP))
base = np.mean(res["전체(기준)"]["lift"])
for name in CONFIGS:
    L = res[name]["lift"]; A = res[name]["auc"]
    d = base - np.mean(L)
    print(f"{name:14s} | 가입률 {np.mean(L):.2f}±{np.std(L):.2f} (Δ{d:+.2f}%p) | AUC {np.mean(A):.3f}±{np.std(A):.3f}")
print("교차항 계수(전체 기준) 안정성: %.3f±%.3f | 특약 계수: %.3f±%.3f" % (
    np.mean(res["전체(기준)"]["xint"]), np.std(res["전체(기준)"]["xint"]),
    np.mean(res["전체(기준)"]["rider_c"]), np.std(res["전체(기준)"]["rider_c"])))

print()
print("=== 2) 가설 E 용량반응 — 특약 부착 상품 개수별 가입률 ===")
pool["r_grp"] = pool["r_cnt"].clip(upper=2).map({0: "0개", 1: "1개", 2: "2개 이상"})
pn = past[past["customer_id"].isin(pool.index)].groupby("customer_id")["product_name"].apply(lambda s: " | ".join(map(str, s)))
pool = pool.join(pn.rename("pnames"))
fam = pool[pool["pnames"].str.contains("M-케어")]
print(f"[M-케어 계열 보유자 n={len(fam)}] 특약 있음/없음:")
for g, sub in fam.groupby(fam["rider"].map({0: "특약없는 M-케어만", 1: "특약 포함"})):
    print(f"  {g:14s} | n={len(sub):6d} | 가입률 {100*sub['label'].mean():.2f}%")
print("E와 F의 얽힘: corr(rider,cnt2)=%.3f, corr(rider,log_pay)=%.3f" % (
    pool["rider"].corr(pool["cnt2"]), pool["rider"].corr(pool["log_pay"])))

print()
print("=== 3) 가설 F 구성 그림자 — 2개 보유자 680명의 저납입 셀은 누구인가 ===")
two = pool[pool["cnt2"] == 1].copy()
two["inv"] = two["pnames"].str.contains("연금|변액|저축")
two["q"] = pd.qcut(two["log_pay"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
for q, sub in two.groupby("q", observed=True):
    print(f"  {q} | n={len(sub):4d} | 투자형 보유 비율 {100*sub['inv'].mean():5.1f}% | 가입률 {100*sub['label'].mean():5.1f}%")
print("[보장형만(투자형 제외) 2개 보유자] 상/하반 납입액:")
safe = two[~two["inv"]]
med = safe["log_pay"].median()
for g, sub in safe.groupby(safe["log_pay"] > med):
    tag = "상반" if g else "하반"
    print(f"  {tag} | n={len(sub):4d} | 가입률 {100*sub['label'].mean():5.1f}%")

# 그림 3패널
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
ax = axes[0]
names5 = list(CONFIGS.keys())
means = [np.mean(res[n]["lift"]) for n in names5]
sds = [np.std(res[n]["lift"]) for n in names5]
bars = ax.bar(range(5), means, yerr=sds, capsize=4,
              color=["#4C72B0" if n == "전체(기준)" else "#CC6677" for n in names5])
ax.axhline(2.92, ls="--", color="#999", lw=1)
ax.annotate("기준선 2.9%", (4.4, 3.2), fontsize=9, color="#777")
ax.set_xticks(range(5)); ax.set_xticklabels(names5, fontsize=9, rotation=12)
ax.set_ylabel("상위 1,000명 실제 가입률(%)")
ax.set_title(f"소거 실험 — 피처 제거 시 하락 ({NREP}회 평균)")
ax2 = axes[1]
g_order = ["0개", "1개", "2개 이상"]
vals = [100 * pool[pool["r_grp"] == g]["label"].mean() for g in g_order]
ns = [len(pool[pool["r_grp"] == g]) for g in g_order]
ax2.bar(range(3), vals, color="#4C72B0")
for i, (v, n) in enumerate(zip(vals, ns)):
    ax2.annotate(f"{v:.1f}%\n(n={n:,})", (i, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
ax2.set_xticks(range(3)); ax2.set_xticklabels(g_order)
ax2.set_ylabel("가입률(%)"); ax2.set_title("가설 E 용량반응 — 특약 상품 개수")
ax3 = axes[2]
qq = ["Q1", "Q2", "Q3", "Q4", "Q5"]
v_all = [100 * two[two["q"] == q]["label"].mean() for q in qq]
n_all = [len(two[two["q"] == q]) for q in qq]
lo_rate = 100 * safe[safe["log_pay"] <= med]["label"].mean()
hi_rate = 100 * safe[safe["log_pay"] > med]["label"].mean()
ax3.bar(range(5), v_all, color="#CC6677", alpha=0.85, label="전체 2개 보유자")
for i, (v, n) in enumerate(zip(v_all, n_all)):
    ax3.annotate(f"{v:.1f}\n(n={n})", (i, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
ax3.axhline((lo_rate + hi_rate) / 2, ls="--", color="#2a9d8f", lw=1.5, label="보장형만(상/하반 평균)")
ax3.annotate(f"보장형만: 하반 {lo_rate:.1f}% / 상반 {hi_rate:.1f}%", (0.02, 0.93), xycoords="axes fraction", fontsize=9, color="#2a9d8f")
ax3.set_xticks(range(5)); ax3.set_xticklabels(qq)
ax3.set_ylabel("가입률(%)"); ax3.set_title("가설 F — 저납입 셀의 구성 점검")
fig.suptitle("가설 E·F 논리 기반 강도 점검 (T=2024, 풀 38,932명)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "가설EF_검증강도.png", dpi=150)
print()
print("그림 저장: output/가설EF_검증강도.png")
