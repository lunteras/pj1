# -*- coding: utf-8 -*-
"""3단계. 기초 모델링 — 로지스틱 회귀 (T=2024년 말 예측 프레임)
라벨 = 2025년 실제 신규 가입 여부 / 평가 = 상위 k명 중 실제 가입률(리프트)"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
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
pool["보유상품수2이상"] = (past.groupby("customer_id").size() >= 2).astype(int)
pool["갱신형보유"] = pool.index.isin(set(past.loc[past["product_name"].str.contains("갱신형", na=False), "customer_id"].unique())).astype(int)
pool["납입액로그"] = np.log1p(past.groupby("customer_id")["transaction_amt"].sum())
pool["상품금액로그"] = np.log1p(past.groupby("customer_id")["product_amt"].sum())

CAT = ["age", "gender", "region", "income_bracket", "occupation_group", "risk_grade", "credit_rating"]
NUM = ["tenure", "납입액로그", "상품금액로그", "보유상품수2이상", "갱신형보유"]

X, y = pool[CAT + NUM], pool["label"]
idx_tr, idx_te = train_test_split(pool.index, test_size=0.3, stratify=y, random_state=42)
Xtr, Xte = X.loc[idx_tr], X.loc[idx_te]
ytr, yte = y.loc[idx_tr], y.loc[idx_te]
print(f"=== 분할(고객 단위, 층화, seed=42) ===\n학습 {len(idx_tr):,}명(양성 {ytr.sum():,}) / 평가 {len(idx_te):,}명(양성 {yte.sum():,}) / 평가 양성률 {yte.mean()*100:.2f}%")

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ("num", StandardScaler(), NUM),
])
pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))])
pipe.fit(Xtr, ytr)
proba = pipe.predict_proba(Xte)[:, 1]

print(f"\n=== 성능 (평가 세트) ===\nROC-AUC {roc_auc_score(yte, proba):.3f} / 평균정밀도(AP) {average_precision_score(yte, proba):.3f} (무작위 기준 {yte.mean():.3f})")

order = np.argsort(-proba)
yte_arr = yte.values
cum_pos = np.cumsum(yte_arr[order])
BASE = yte.mean() * 100

# 규칙(보유상품수2이상) 비교 — 평가 세트 내
rule_mask = Xte["보유상품수2이상"].values == 1
rule_n = int(rule_mask.sum()); rule_pos = int(yte_arr[rule_mask].sum())
rule_rate = rule_pos / rule_n * 100
rule_cap = rule_pos / yte.sum() * 100
print(f"\n=== 상위 k명 리프트 (평가 세트 {len(idx_te):,}명, 양성 {int(yte.sum()):,}명, 기준선 {BASE:.1f}%) ===")
print(f"{'k':>7} | {'실제 가입자':>9} | {'가입률(%)':>9} | {'배수':>6} | {'양성 포착률(%)':>12}")
for k in [500, 1000, 2000]:
    pos = int(cum_pos[k - 1]); rate = pos / k * 100
    print(f"{k:>7,} | {pos:>9,} | {rate:>9.1f} | {rate/BASE:>5.1f}x | {pos/yte.sum()*100:>11.1f}")
print(f"\n[규칙 비교] 보유상품수2이상 규칙: 후보 {rule_n:,}명 / 가입자 {rule_pos}명 / 가입률 {rule_rate:.1f}% ({rule_rate/BASE:.1f}x) / 포착률 {rule_cap:.1f}%")
mk = rule_n
mpos = int(cum_pos[mk - 1]); mrate = mpos / mk * 100
print(f"[같은 규모 비교] 모델 상위 {mk:,}명: 가입률 {mrate:.1f}% ({mrate/BASE:.1f}x) / 포착률 {mpos/yte.sum()*100:.1f}%  → 규칙 {rule_rate:.1f}% 대비 {mrate-rule_rate:+.1f}%p")

# ---- 그림 1: 리프트(포착률) 곡선 ----
ks_pct = np.arange(1, 31)  # 상위 1~30%
ks = (ks_pct / 100 * len(yte_arr)).astype(int)
capture = cum_pos[ks - 1] / yte.sum() * 100
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.plot(ks_pct, capture, color="#4C72B0", lw=2, label="모델 (상위 k% 접촉 시 잡는 실제 가입자 비율)")
ax.plot(ks_pct, ks_pct, color="gray", ls=":", lw=1.2, label="무작위 접촉 (포착률 = 접촉 비율)")
ax.axvline(rule_n / len(yte_arr) * 100, color="darkorange", ls="--", lw=1.2)
ax.scatter([rule_n / len(yte_arr) * 100], [rule_cap], color="darkorange", zorder=5)
ax.annotate(f"규칙(보유상품수2+)\n접촉 {rule_n/len(yte_arr)*100:.1f}% → 포착 {rule_cap:.1f}%",
            (rule_n / len(yte_arr) * 100, rule_cap), textcoords="offset points", xytext=(12, -34), fontsize=9, color="darkorange")
for pos, (k, kp) in enumerate(zip(ks, ks_pct)):
    if kp in (5, 10, 20, 30):
        ax.annotate(f"{capture[pos]:.0f}%", (kp, capture[pos]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color="#4C72B0")
ax.set_xlabel("상담 접촉 비율(%) — 모델 확률 상위부터")
ax.set_ylabel("실제 가입자 포착률(%)")
ax.set_title(f"3단계 기초 모델 리프트 곡선 (평가 세트 {len(idx_te):,}명, 양성 {int(yte.sum()):,}명)\n전체 가입률 {BASE:.1f}% — 전원 접촉 시 포착률 100%, 무작위는 대각선", fontsize=11)
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 31); ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig(f"{OUT}\\모델_리프트곡선.png", dpi=150)
plt.close(fig)

# ---- 그림 2: 계수 상위/하위 (설명) ----
feat = pipe.named_steps["pre"].get_feature_names_out()
coef = pipe.named_steps["clf"].coef_[0]
cd = pd.DataFrame({"피처": feat, "계수": coef})
cd["피처"] = cd["피처"].str.replace(r"^(cat__|num__)", "", regex=True)
top = pd.concat([cd.nlargest(10, "계수"), cd.nsmallest(10, "계수")]).sort_values("계수")
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(top["피처"], top["계수"], color=np.where(top["계수"] > 0, "#4C72B0", "#C44E52"))
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("로지스틱 계수 (표준화 후 — 클수록 가입 확률 올림)")
ax.set_title("기초 모델 기여도 상위 (양수 = 가입 쪽, 음수 = 비가입 쪽)\n계수는 방향 참고용 — 최종 판정은 리프트 곡선", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}\\모델_계수_상위하위.png", dpi=150)
plt.close(fig)

print("\n=== 계수 상위 (양수 10 / 음수 10) ===")
print(cd.reindex(cd["계수"].abs().sort_values(ascending=False).index).head(20).round(3).to_string(index=False))
print("\n그림 저장: output/모델_리프트곡선.png, output/모델_계수_상위하위.png")
