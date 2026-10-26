# -*- coding: utf-8 -*-
"""
'기타' 편향 점검 — T=2024년 말 프레임
목적: 대체값 '기타'를 EDA 집계에서 제외한 4개 열(직업·병력·지역·신용)에서,
      '기타' 집단의 가입률이 전체(2.9%)나 실제 범주들과 크게 달라
      제외 집계에 편향을 준 것이 없는지 확인.
프레임: transaction_start_year <= 2024인 보험 고객 38,932명, 타겟 = 2025년 가입 여부
판정: '기타' 가입률이 전체와 ±1%p 이내이면 편향 없음.
산출: 콘솔 표 + output/기타편향_점검.png
"""
import io, sys, zipfile, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일"
ZPATH = None
for _base, _dirs, _files in os.walk(ROOT):
    for f in _files:
        if f.startswith("TL_2") and f.endswith(".zip"):
            ZPATH = os.path.join(_base, f)
            break
    if ZPATH:
        break

with zipfile.ZipFile(ZPATH) as z:
    name = [n for n in z.namelist() if n.endswith(".csv")][0]
    with z.open(name) as f:
        df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)

ins = df[df["customer_type"] == "보험"].copy()
cnt = ins.groupby("customer_id")["transaction_start_year"].agg(
    min_year="min", has_after=lambda s: int((s > 2024).any())
).reset_index()
pool = cnt[cnt["min_year"] <= 2024].drop_duplicates("customer_id")
pool = pool.merge(ins.drop_duplicates("customer_id")[[
    "customer_id", "occupation_group", "disease_history", "region", "credit_rating"]], on="customer_id")
n = len(pool)
base = pool["has_after"].mean() * 100
print(f"풀 {n:,}명 · 전체 가입률 {base:.1f}%")

cols = {"직업": "occupation_group", "병력": "disease_history", "지역": "region", "신용": "credit_rating"}
rows = []
for label, col in cols.items():
    etc = pool[pool[col] == "기타"]
    real = pool[pool[col] != "기타"]
    rates = real.groupby(col)["has_after"].mean() * 100
    rows.append((label, len(etc), etc["has_after"].mean() * 100,
                 rates.min(), rates.max(), (etc["has_after"].mean() * 100) - base))

fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.axis("off")
tbl = ax.table(cellText=[[f"{c[0]}", f"{c[1]:,}", f"{c[2]:.2f}%", f"{c[3]:.1f}~{c[4]:.1f}%",
                         f"{c[5]:+.2f}%p", "편향 없음" if abs(c[5]) < 1 else "편향 의심"] for c in rows],
               colLabels=["열", "기타 n", "기타 가입률", "실범주 범위", "전체 대비", "판정"],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.5)
plt.tight_layout()
out = os.path.join(ROOT, "ABA_1st_proj_시작파일", "output", "기타편향_점검.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"그림 저장: {out}")

print()
print("=== '기타' 집단별 점검 ===")
print(f"{'열':<6}{'기타 n':>8}{'기타 가입률':>10}{'실범주 범위':>16}{'전체대비':>10}")
for r in rows:
    print(f"{r[0]:<6}{r[1]:>8,}{r[2]:>9.2f}%{r[3]:>7.1f}~{r[4]:.1f}%{r[5]:>+9.2f}%p")
print()
print("판정: |전체대비| < 1%p → 편향 없음 / >= 1%p → 편향 의심")
