# -*- coding: utf-8 -*-
"""새 가설 E(특약 보유)·F(보유수×납입액 교차) — 전체 형식 검증
E: 주보험+특약 상품 보유 고객이 주보험만 고객보다 5배 이상 잘 가입하는가
F: 2개 보유 고객의 가입률이 납입액과 교차해 급등하는가(1개 보유는 평평)
T=2024 메인, T=2023 교차. 그림 2장 저장."""
import io, sys, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

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
cust = ins.drop_duplicates("customer_id").set_index("customer_id")
pools = {}
for T in [2024, 2023]:
    past = ins[ins["transaction_start_year"] <= T]
    past_ids = set(past["customer_id"].unique())
    fut_ids = set(ins.loc[ins["transaction_start_year"] > T, "customer_id"].unique())
    pool = cust[cust.index.isin(past_ids)].copy()
    pool["label"] = pool.index.isin(fut_ids).astype(int)
    prod = past.groupby("customer_id").agg(
        product_cnt=("product_name", "size"),
        pay_sum=("transaction_amt", "sum"),
        names=("product_name", lambda s: " | ".join(sorted(set(s)))),
        kws=("transaction_keyword", lambda s: " | ".join(sorted(set(map(str, s))))),
    )
    pool = pool.join(prod)
    # E 그룹 정의: 키워드 문자열에 '주보험'과 '특약'이 같이 있으면 특약 부착 주보험
    def e_group(row):
        parts = str(row["kws"]).split(" | ")
        has_rider = any(("주보험" in k) and ("특약" in k) for k in parts)
        has_main = any(("주보험" in k) and ("특약" not in k) for k in parts)
        has_ridonly = any(("주보험" not in k) and ("특약" in k) for k in parts)
        if has_rider: return "주보험+특약"
        if has_main: return "주보험만"
        if has_ridonly: return "특약만"
        return "기타"
    pool["E그룹"] = pool.apply(e_group, axis=1)
    # 투자형(변액/연금) 상품 보유 플래그 — 상품 이름 기반
    pool["투자형보유"] = pool["names"].apply(lambda s: any(("변액" in n or "연금" in n) for n in str(s).split(" | ")))
    pools[T] = pool
    print(f"T={T}: 풀 {len(pool):,} / 양성 {int(pool['label'].sum()):,} / {pool['label'].mean()*100:.1f}%")

def rate_tbl(df, by):
    g = df.groupby(by)["label"].agg(["count", "sum", "mean"])
    g.columns = ["고객 수", "가입자", "가입률(%)"]
    g["가입률(%)"] = (g["가입자"] / g["고객 수"] * 100).round(1)
    return g

print("\n" + "#" * 62)
print("# 가설 E — 특약 보유 = 관계가 깊은 고객 (5배 이상)")
print("#" * 62)
for T in [2024, 2023]:
    p = pools[T]
    print(f"\n[T={T}] E그룹별 가입률")
    g = rate_tbl(p, "E그룹")
    order = [x for x in ["주보험+특약", "주보험만", "특약만", "기타"] if x in g.index]
    print(g.reindex(order).to_string())
    small = [f"{i}={int(r['고객 수'])}" for i, r in g.reindex(order).iterrows() if r["고객 수"] < 30]
    print("30미만 셀: " + (", ".join(small) or "없음"))
    # 그림자 확인 1: 보유상품수 조각 안에서
    one = p[p["product_cnt"] == 1]
    print(f"[T={T}] 1개 보유자 조각 안에서 (n={len(one):,})")
    g1 = rate_tbl(one, "E그룹")
    print(g1.reindex([x for x in order if x in g1.index]).to_string())
    # 그림자 크기: E그룹별 2개 이상 보유 비율
    comp = p.groupby("E그룹").agg(보유2이상비율=("product_cnt", lambda s: (s >= 2).mean() * 100),
                                  평균보유수=("product_cnt", "mean")).round(1)
    print("[그림자 크기] E그룹별 2개 이상 보유 비율(%), 평균 보유수")
    print(comp.reindex(order).to_string())

# E 함정 3: 특약만 고객은 무슨 상품을 보유하는가 (0.0% 수수께끼)
p = pools[2024]
rid_only_rows = ins[ins["customer_id"].isin(p.index) & (ins["transaction_start_year"] <= 2024)]
kw_only = rid_only_rows[rid_only_rows["transaction_keyword"].astype(str).str.contains("특약", na=False) &
                        ~rid_only_rows["transaction_keyword"].astype(str).str.contains("주보험", na=False)]
print("\n[특약만 행의 상품 이름 상위 6] (T=2024)")
print(kw_only["product_name"].value_counts().head(6).to_string())

print("\n" + "#" * 62)
print("# 가설 F — 보유수×납입액 교차: 2개 보유자에서 급등, 1개는 평평")
print("#" * 62)
p = pools[2024].copy()
p["납입분위"] = pd.qcut(p["pay_sum"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
p["보유"] = np.where(p["product_cnt"] >= 2, "2개 이상", "1개")
ct = pd.crosstab(p["보유"], p["납입분위"], values=p["label"], aggfunc="mean") * 100
n = pd.crosstab(p["보유"], p["납입분위"])
print("[T=2024] 보유수×납입액5분위 가입률(%)")
print(ct.round(1).to_string())
print("셀 크기(고객 수):")
print(n.to_string())
print("30미만 셀: " + (", ".join(f"{i}·{c}={int(n.loc[i, c])}" for i in n.index for c in n.columns if n.loc[i, c] < 30) or "없음"))
# 납입액 자체의 비순환성: 보유수별 납입액 중앙값
med = p.groupby("보유")["pay_sum"].median() / 1e4
print(f"[비순환성 확인] 납입액 중앙값(만원, T까지 합계): " + " / ".join(f"{i} {v:.0f}" for i, v in med.items()))
# T=2023 교차: 2개 보유자는 n=77이라 중앙값 분할
q = pools[2023].copy()
q["보유"] = np.where(q["product_cnt"] >= 2, "2개 이상", "1개")
for grp_name, sub in q.groupby("보유"):
    sub = sub.copy()
    if grp_name == "2개 이상":
        sub["분위"] = pd.qcut(sub["pay_sum"], 2, labels=["하반", "상반"])
    else:
        sub["분위"] = pd.qcut(sub["pay_sum"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    g = rate_tbl(sub, "분위")
    small = [f"{i}={int(r['고객 수'])}" for i, r in g.iterrows() if r["고객 수"] < 30]
    print(f"[T=2023] {grp_name}: {g.to_dict('index')} / 30미만 셀: " + (", ".join(small) or "없음"))

# ---------- 그림 E ----------
p = pools[2024]
order = ["주보험+특약", "주보험만", "특약만"]
g_all = rate_tbl(p, "E그룹").reindex(order)
one = p[p["product_cnt"] == 1]
g_one_raw = rate_tbl(one, "E그룹")
g_one = g_one_raw.reindex([x for x in order if x in g_one_raw.index])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
bars = ax.bar(g_all.index, g_all["가입률(%)"], color=["#C44E52", "#4C72B0", "#937860"], width=0.5)
for b, v in zip(bars, g_all["가입률(%)"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v}%", ha="center", fontsize=10)
ax.axhline(2.9, color="gray", ls="--", lw=1)
ax.text(2.45, 2.95, "전체 2.9%", fontsize=8, color="gray", ha="right")
ax.set_title("가설 E — E그룹별 가입률 (T=2024, 전체)", fontsize=11)
ax.set_ylabel("가입률(%)")
ax = axes[1]
bars = ax.bar(g_one.index, g_one["가입률(%)"], color=["#C44E52", "#4C72B0"], width=0.5)
for b, v in zip(bars, g_one["가입률(%)"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v}%", ha="center", fontsize=10)
ax.axhline(2.9, color="gray", ls="--", lw=1)
ax.set_title("1개 보유자 조각 안에서 — 보유수 그림자 제거 비교", fontsize=11)
ax.set_ylabel("가입률(%)")
fig.suptitle("새 가설 E — 특약이 붙은 상품 보유 고객이 추가 가입을 잘하는가", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}\\새가설_E_특약보유.png", dpi=150); plt.close(fig)

# ---------- 그림 F ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
xs = ["Q1", "Q2", "Q3", "Q4", "Q5"]
ax = axes[0]
for grp, color in [("1개", "#4C72B0"), ("2개 이상", "#C44E52")]:
    vals = [ct.loc[grp, x] if x in ct.columns else np.nan for x in xs]
    ax.plot(xs, vals, marker="o", color=color, label=grp)
ax.set_title("가설 F — 보유수×납입액 교차 (T=2024)\n2개 보유 선이 급등하고 1개 선은 평평해야 지지", fontsize=10)
ax.set_xlabel("납입액 5분위"); ax.set_ylabel("가입률(%)"); ax.legend()
ax = axes[1]
qq = pools[2023].copy()
qq["보유"] = np.where(qq["product_cnt"] >= 2, "2개 이상", "1개")
sub1 = qq[qq["보유"] == "1개"].copy()
sub1["분위"] = pd.qcut(sub1["pay_sum"], 5, labels=xs)
g1 = sub1.groupby("분위")["label"].mean() * 100
ax.plot(xs, [g1.get(x) for x in xs], marker="o", color="#4C72B0", label="1개")
sub2 = qq[qq["보유"] == "2개 이상"].copy()
sub2["분위"] = pd.qcut(sub2["pay_sum"], 2, labels=["하반", "상반"])
g2 = sub2.groupby("분위")["label"].mean() * 100
ax.plot([0.1, 0.9], [g2.get("하반"), g2.get("상반")], marker="s", color="#C44E52", label="2개 이상 (하반/상반)")
ax.set_title("T=2023 교차 (2개 보유는 n=77, 중앙값 분할 — 참고용)", fontsize=10)
ax.set_xlabel("납입액"); ax.set_ylabel("가입률(%)"); ax.legend()
fig.suptitle("새 가설 F — 확장은 고금액 반복 고객에서 먼저 일어나는가", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(f"{OUT}\\새가설_F_보유수x납입액.png", dpi=150); plt.close(fig)

print("\n그림 저장: output/새가설_E_특약보유.png, output/새가설_F_보유수x납입액.png")
