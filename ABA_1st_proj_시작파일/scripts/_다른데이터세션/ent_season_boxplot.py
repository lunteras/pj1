import sys
import os
import pandas as pd
from matplotlib import font_manager, pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

ent = pd.read_csv(r"[엔터] Netflix Top 10 Weekly Dataset/all-weeks-global.csv", parse_dates=["week"])
ent["st"] = ent["season_title"].fillna("")
ent["grp"] = "비시즌제(영화·단발)"
ent.loc[ent["category"].str.startswith("TV")
        & ent["st"].str.contains(r"(?:Season|Part|Volume|Vol\.?)\s*\d", regex=True), "grp"] = "시즌제 TV"

def eok(v):
    return f"{v/1e8:.1f}억" if v >= 1e8 else f"{v/1e4:,.0f}만"

summary = ent.groupby("grp")["weekly_hours_viewed"].agg(["count", "mean", "median", "max"])
summary["평균대중앙"] = (summary["mean"] / summary["median"]).round(1)
print("=== 집단별 요약 (시간 단위) ===")
print(summary.to_string())
q = ent.groupby("grp")["weekly_hours_viewed"].quantile([0.25, 0.75]).unstack()
print("\n사분위수:")
print(q.to_string())
print("\n=== 집단별 시청시간 상위 3행 ===")
for g in ent["grp"].unique():
    top = ent[ent["grp"] == g].nlargest(3, "weekly_hours_viewed")[["week", "show_title", "weekly_hours_viewed"]]
    print(f"[{g}]")
    print(top.to_string(index=False))

names = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

os.makedirs("output", exist_ok=True)

groups = ["시즌제 TV", "비시즌제(영화·단발)"]
data = [ent.loc[ent["grp"] == g, "weekly_hours_viewed"] for g in groups]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
bp = ax.boxplot(data, tick_labels=["시즌제 TV", "비시즌제\n(영화·단발)"], patch_artist=True,
                showmeans=True, widths=0.5,
                meanprops=dict(marker="D", markerfacecolor="red", markeredgecolor="red", markersize=7),
                medianprops=dict(color="black", linewidth=2),
                boxprops=dict(linewidth=1.2),
                flierprops=dict(marker="o", markersize=3, alpha=0.35))
for patch, color in zip(bp["boxes"], ["#7fb3d5", "#f5b041"]):
    patch.set_facecolor(color)

ax.set_yscale("log")
ax.set_ylim(4e5, 1.4e9)
ax.set_yticks([1e6, 1e7, 1e8])
ax.set_yticklabels(["100만", "1,000만", "1억"])
ax.minorticks_off()
ax.set_ylabel("주간 시청시간 (시간, 로그 눈금)")
ax.set_title("시즌제·비시즌제 주간 시청시간 분포\n(빨간 마름모 = 평균, 검은 굵은 선 = 중앙값)")
ax.grid(axis="y", alpha=0.3)

for i, g in enumerate(groups, start=1):
    sub = ent.loc[ent["grp"] == g, "weekly_hours_viewed"]
    ax.annotate(f"평균 {eok(sub.mean())}", xy=(i + 0.28, sub.mean()), fontsize=9, color="red")
    ax.annotate(f"중앙값 {eok(sub.median())}", xy=(i + 0.28, sub.median() * 0.85), fontsize=9, color="black")

plt.tight_layout()
path = os.path.join("output", "엔터_시즌제_시청시간_상자그림.png")
plt.savefig(path, dpi=150)
print("\n그림 저장:", os.path.abspath(path))
