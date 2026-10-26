import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
from matplotlib import font_manager, pyplot as plt

names = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
print("사용 글꼴:", plt.rcParams["font.family"])

fin = pd.read_csv(
    r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip",
    encoding="utf-8-sig",
)

sec = fin[fin["customer_type"] == "증권"]
cust = sec.drop_duplicates(subset="customer_id")

order = ["1000만원미만", "3000만원미만", "5000만원미만", "10000만원미만", "10000만원이상", "30000만원미만"]
order_label = ["1천만원\n미만", "3천만원\n미만", "5천만원\n미만", "1억원\n미만", "1억원\n이상", "3억원\n미만"]

ct = pd.crosstab(cust["income_bracket"], cust["investment_propensity"]).reindex(order)
pct = (ct.div(ct.sum(axis=1), axis=0) * 100)
counts = ct.sum(axis=1)

series = [
    ("안정형", pct["안정형"].values, "#9ecae1"),
    ("안정추구형", pct["안정추구형"].values, "#4292c6"),
    ("안정형+안정추구형", (pct["안정형"] + pct["안정추구형"]).values, "#08519c"),
    ("공격투자형", pct["공격투자형"].values, "#d9534f"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.1, 1]})
x = range(len(order))
n = len(series)
w = 0.8 / n

for i, (name, vals, color) in enumerate(series):
    pos = [xi - 0.4 + w * (i + 0.5) for xi in x]
    bars = ax1.bar(pos, vals, width=w, label=name, color=color)
    ax1.bar_label(bars, fmt="%.1f", fontsize=7.5, padding=1)

ax1.set_xticks(list(x))
ax1.set_xticklabels(order_label, fontsize=9)
ax1.set_ylabel("고객 비율(%)")
ax1.set_xlabel("소득 구간")
ax1.set_title("소득 구간별 투자성향 비율 (증권 고객, 고객 단위)", fontsize=12)
ax1.legend(fontsize=9, ncol=2)
ax1.grid(axis="y", alpha=0.3)
ax1.set_axisbelow(True)
ax1.set_ylim(0, 42)

bars2 = ax2.bar(order_label, counts.values, color="#8c8c8c")
ax2.bar_label(bars2, labels=[f"{v:,}명" for v in counts.values], fontsize=8, padding=1)
ax2.set_ylabel("고객 수(명)")
ax2.set_xlabel("소득 구간")
ax2.set_title("소득 구간별 고객 수", fontsize=12)
ax2.grid(axis="y", alpha=0.3)
ax2.set_axisbelow(True)
ax2.tick_params(axis="x", labelsize=8)
ax2.set_ylim(0, 118000)

plt.tight_layout()
out = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\output\금융_소득구간_투자성향_비율_막대.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("저장 완료:", out)