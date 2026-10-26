import sys, re
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from matplotlib import font_manager, pyplot as plt

names = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

fin = pd.read_csv("[금융] 금융상품·서비스 및 소비자 특성 데이터/Training/02.라벨링데이터/TL_2.소비자.zip", encoding="utf-8-sig")
ins = fin[fin["customer_type"] == "보험"].copy()
blanks = ["미수집", "미답변", "-", "기타"]
sub = ins[~(ins["credit_rating"].isin(blanks) | ins["credit_rating"].isna())].copy()
sub["transaction_amt"] = pd.to_numeric(sub["transaction_amt"], errors="coerce")
sub = sub.dropna(subset=["transaction_amt"])

def band_key(s):
    m = re.search(r"(\d{3})", str(s))
    return int(m.group(1)) if m else -1

cust = sub.groupby("customer_id").agg(total_amt=("transaction_amt", "sum"), band=("credit_rating", "first"))
bands = sorted(cust["band"].unique(), key=band_key, reverse=True)
data = [cust.loc[cust["band"] == b, "total_amt"] / 1e4 for b in bands]
overall_median = cust["total_amt"].median() / 1e4

fig, ax = plt.subplots(figsize=(9, 6))
ax.boxplot(
    data,
    showmeans=True,
    meanprops={"marker": "D", "markerfacecolor": "darkorange", "markeredgecolor": "darkorange", "markersize": 7},
    medianprops={"color": "black", "linewidth": 1.5},
    flierprops={"markersize": 2, "alpha": 0.3},
)
ax.set_xticks(range(1, len(bands) + 1))
ax.set_xticklabels(bands)
ax.set_yscale("log")
ax.axhline(overall_median, color="gray", linestyle="--", linewidth=1)
ax.text(len(bands) + 0.42, overall_median, f"전체 중앙값 {overall_median:.0f}만원", va="center", fontsize=9, color="gray")
ax.set_title("신용점수대별 보험 고객당 총 납입액 상자 그림 (고객 74,065명)", fontsize=14)
ax.set_xlabel("신용점수대")
ax.set_ylabel("고객당 총 납입액 (만원, 로그 축)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out = "output/금융_신용점수대_고객총액_상자그림.png"
plt.savefig(out, dpi=150)
print("저장:", out)
for b, d in zip(bands, data):
    print(f"{b} | n={len(d):,} | 중앙값 {d.median():.1f}만원 | 평균 {d.mean():.1f}만원 | 평균/중앙값 {d.mean()/d.median():.2f}")
