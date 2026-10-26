import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ent = pd.read_csv(r"[엔터] Netflix Top 10 Weekly Dataset/all-weeks-global.csv", parse_dates=["week"])

print("=== 행/열 수 ===")
print(ent.shape)

print("\n=== 열 이름과 dtype ===")
for c in ent.columns:
    print(f"{c}: {ent[c].dtype}")

print("\n=== 결측 개수 ===")
print(ent.isna().sum().to_dict())

print("\n=== 고유값 개수 ===")
print(ent.nunique().to_dict())

print("\n=== category 분포 ===")
print(ent["category"].value_counts().to_dict())

print("\n=== week 범위 ===")
print(ent["week"].min(), "~", ent["week"].max(), "/ 주 수:", ent["week"].nunique())

print("\n=== 작품 수 (category+show_title+season_title 묶음 기준) ===")
print(ent.groupby(["category", "show_title", "season_title"]).ngroups)

print("\n=== TV season_title 예시 8개 ===")
tv = ent.loc[ent["category"].str.startswith("TV"), "season_title"].unique()[:8]
print(list(tv))

print("\n=== 숫자 열 요약 ===")
print(ent[["weekly_rank", "weekly_hours_viewed", "cumulative_weeks_in_top_10"]].describe().round(1))

print("\n=== 작품별 Top 10 잔존 주 수(최대 누적주) 분포 ===")
per_show = ent.groupby(["category", "show_title", "season_title"])["cumulative_weeks_in_top_10"].max()
print(per_show.describe().round(1))

print("\n=== 시청시간 상위 5행 ===")
top5 = ent.nlargest(5, "weekly_hours_viewed")[["week", "category", "show_title", "weekly_hours_viewed"]]
print(top5.to_string(index=False))
