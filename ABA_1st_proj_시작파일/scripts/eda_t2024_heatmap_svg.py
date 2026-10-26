# -*- coding: utf-8 -*-
"""T=2024 점검 — 대시보드용 상관 히트맵 SVG 생성 (범주형 크라메르 V 9x9)

eda_t2024_heatmap.py와 동일한 풀(T=2024년 말 보유 고객 38,932명)에서
범주형 변수 간 크라메르 V 행렬을 계산해, 최종발표_대시보드.html 상관관계
패널에 붙여 넣을 SVG 조각을 output/상관히트맵_발표용.svg로 저장한다.

기존 검증 수치와의 일치 점검을 함께 수행한다(불일치 시 FAIL 표시):
  - 연령대×결혼 0.642 / 직업군×위험등급 0.572 / 성별×직업 0.490
  - 라벨-숫자형 피어슨: 보유상품수 +0.070 / 납입액(로그) +0.050 /
    상품금액(로그) +0.020 / 경과연수 +0.004
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

SRC = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"
OUT = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\output"

fin = pd.read_csv(SRC, encoding="utf-8-sig")
print(f"원본 행 수 {len(fin):,} / 열 수 {len(fin.columns)}")
ins = fin[fin["customer_type"] == "보험"]
T = 2024
past = ins[ins["transaction_start_year"] <= T]
cust = ins.drop_duplicates("customer_id").set_index("customer_id")
past_ids = set(past["customer_id"].unique())
pool = cust[cust.index.isin(past_ids)].copy()
pool["label"] = pool.index.isin(set(ins.loc[ins["transaction_start_year"] > T, "customer_id"].unique())).astype(int)
print(f"T={T} 풀 고객 {len(pool):,}명 / 정답(2025 신규 가입) {int(pool['label'].sum()):,}명")

def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n = ct.values.sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

cats = ["라벨", "연령대", "성별", "지역", "소득구간", "직업군", "위험등급", "신용점수대", "결혼"]
cols = ["label", "age", "gender", "region", "income_bracket", "occupation_group",
        "risk_grade", "credit_rating", "marital_status"]
cv = pd.DataFrame(index=cats, columns=cats, dtype=float)
for i in range(len(cols)):
    for j in range(len(cols)):
        cv.iloc[i, j] = cramers_v(pool[cols[i]], pool[cols[j]])

# ---- 기존 검증 수치와의 일치 점검 ----
checks = [
    ("연령대×결혼", cv.loc["연령대", "결혼"], 0.642),
    ("직업군×위험등급", cv.loc["직업군", "위험등급"], 0.572),
    ("성별×직업군", cv.loc["성별", "직업군"], 0.490),
]
lab_v = {c: cv.loc["라벨", c] for c in cats[1:]}
print("\n[크라메르 V 일치 점검]")
ok = True
for name, got, want in checks:
    good = abs(got - want) <= 0.001
    ok &= good
    print(f"  {name}: {got:.3f} (기존 {want}) {'PASS' if good else 'FAIL'}")
print(f"  라벨-범주형 범위: {min(lab_v.values()):.3f}~{max(lab_v.values()):.3f} (기존 기록 0.02~0.05)")

first_year = past.groupby("customer_id")["transaction_start_year"].min()
pool["tenure"] = T - pool.index.map(first_year)
pool["보유상품수"] = past.groupby("customer_id").size()
pool["납입액합계"] = np.log1p(past.groupby("customer_id")["transaction_amt"].sum())
pool["상품금액합계"] = np.log1p(past.groupby("customer_id")["product_amt"].sum())
pear = pool[["label", "tenure", "보유상품수", "납입액합계", "상품금액합계"]].corr()["label"]
pear_checks = [("보유상품수", pear["보유상품수"], 0.070), ("납입액(로그)", pear["납입액합계"], 0.050),
               ("상품금액(로그)", pear["상품금액합계"], 0.020), ("경과연수", pear["tenure"], 0.004)]
print("[피어슨 일치 점검]")
for name, got, want in pear_checks:
    good = abs(got - want) <= 0.001
    ok &= good
    print(f"  라벨×{name}: {got:+.3f} (기존 {want:+.3f}) {'PASS' if good else 'FAIL'}")
print("\n종합:", "PASS" if ok else "FAIL — 수치 불일치, 대시보드 반영 금지")
if not ok:
    sys.exit(1)

# ---- SVG 생성 (범주형 크라메르 V 9x9 히트맵) ----
CELL, GAP, LABELW, TOPH, PADL = 44, 2, 96, 88, 8
W = PADL + LABELW + len(cats) * CELL + 8
H = TOPH + len(cats) * CELL + 58
P = []
P.append(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="범주형 변수 9개와 라벨 사이의 크라메르 V 히트맵 — 변수끼리는 연령대×결혼 0.642 등 짙은 셀이 있지만, 라벨 행은 전부 0.05 이하로 얇다">')
for j, c in enumerate(cats):
    cx = PADL + LABELW + j * CELL + CELL / 2
    cls = ' class="c-red"' if c == "라벨" else ' class="c-mut"'
    P.append(f'  <text{cls} x="{cx + 4:.1f}" y="{TOPH - 8}" font-size="12" font-weight="{700 if c == "라벨" else 400}" text-anchor="end" transform="rotate(-45 {cx + 4:.1f} {TOPH - 8})">{c}</text>')
for i, r in enumerate(cats):
    y = TOPH + i * CELL
    cls = ' class="c-red"' if r == "라벨" else ' class="c-mut"'
    P.append(f'  <text{cls} x="{PADL + LABELW - 8}" y="{y + CELL / 2 + 4.5:.1f}" font-size="12.5" font-weight="{700 if r == "라벨" else 400}" text-anchor="end">{r}</text>')
    for j, c in enumerate(cats):
        x = PADL + LABELW + j * CELL
        v = float(cv.iloc[i, j])
        if i == j:
            P.append(f'    <rect class="hm-diag" x="{x + GAP}" y="{y + GAP}" width="{CELL - 2 * GAP}" height="{CELL - 2 * GAP}"/>')
            continue
        on = v >= 0.45
        tc = "hm-on" if on else "hm-off"
        fw = ' font-weight="700"' if v >= 0.4 else ""
        P.append(f'    <rect class="hm-cell" x="{x + GAP}" y="{y + GAP}" width="{CELL - 2 * GAP}" height="{CELL - 2 * GAP}" fill-opacity="{v:.2f}"/>')
        P.append(f'    <text class="{tc}" x="{x + CELL / 2:.1f}" y="{y + CELL / 2 + 3.5:.1f}" font-size="10" text-anchor="middle"{fw}>{v:.2f}</text>')
# 정답(라벨) 행·열 강조 테두리
gx, gy = PADL + LABELW, TOPH
P.append(f'  <rect class="hm-outline" x="{gx}" y="{gy}" width="{len(cats) * CELL}" height="{CELL}"/>')
P.append(f'  <rect class="hm-outline" x="{gx}" y="{gy}" width="{CELL}" height="{len(cats) * CELL}"/>')
P.append(f'  <line class="l-axis" x1="{gx}" y1="{TOPH}" x2="{W - 6}" y2="{TOPH}"/>')
P.append(f'  <line class="l-axis" x1="{gx}" y1="{TOPH}" x2="{gx}" y2="{H - 50}"/>')
P.append('  <text class="c-mut" x="8" y="' + str(H - 20) + '" font-size="11.5">진한 셀일수록 두 변수가 많이 겹침(크라메르 V, 0~1) · 대각선(같은 변수끼리)은 회색 · 빨간 점선 테두리 행·열 = 정답(추가 가입)과의 연관</text>')
P.append('  <text class="c-mut" x="8" y="' + str(H - 5) + '" font-size="11.5">테두리 안 셀은 전부 0.05 이하 — 개별 속성만으로는 정답이 갈라지지 않는다</text>')
P.append('</svg>')

svg_path = OUT + r"\상관히트맵_발표용.svg"
with open(svg_path, "w", encoding="utf-8") as f:
    f.write("\n".join(P))
print(f"\nSVG 저장: {svg_path}")
print("\n[라벨 행 값 — 그대로 대시보드 노트에 쓸 수 있는 형태]")
print(" / ".join(f"{c} {lab_v[c]:.3f}" for c in cats[1:]))
print("\n[행렬 전체]")
print(cv.round(3).to_string())
