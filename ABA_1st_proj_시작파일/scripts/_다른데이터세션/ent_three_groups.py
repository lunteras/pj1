import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ent = pd.read_csv(r"[엔터] Netflix Top 10 Weekly Dataset/all-weeks-global.csv", parse_dates=["week"])
ent["st"] = ent["season_title"].fillna("")
ent["key"] = list(zip(ent["category"], ent["show_title"], ent["st"]))
pair_set = set(zip(ent["key"], ent["week"]))
ent["survives"] = [(k, w + pd.Timedelta(days=7)) in pair_set for k, w in zip(ent["key"], ent["week"])]
last = ent["week"].max()

tv = ent["category"].str.startswith("TV")
is_season = tv & ent["st"].str.contains(r"(?:Season|Part|Volume|Vol\.?)\s*\d", regex=True)
ent["grp3"] = "영화"
ent.loc[tv, "grp3"] = "단발 TV"
ent.loc[is_season, "grp3"] = "시즌제 TV"

sub = ent[ent["week"] != last]
t = sub.groupby("grp3")["survives"].agg(행_수="count", 잔존비율=lambda s: round(100 * s.mean(), 1))
print("=== 다음 주 잔존 비율 (마지막 주 40행 제외) ===")
print(t.to_string())
print("\n합계 검증: 전체", len(ent), "= 표 합계", t["행_수"].sum(), "+ 마지막 주", (ent["week"] == last).sum())

t2 = ent.groupby("grp3")["weekly_hours_viewed"].agg(행_수="count", 중앙값_만시간="median", 평균_만시간="mean", 최대_억시간="max")
t2["중앙값_만시간"] = (t2["중앙값_만시간"] / 1e4).round(0)
t2["평균_만시간"] = (t2["평균_만시간"] / 1e4).round(0)
t2["최대_억시간"] = (t2["최대_억시간"] / 1e8).round(2)
print("\n=== 참고: 시청시간 (전체 1,160행) ===")
print(t2.to_string())

dtv = ent[ent["grp3"] == "단발 TV"]
print("\n단발 TV 중 Limited Series 행 수:", dtv["st"].str.contains("Limited").sum(), "/", len(dtv))
print("\n=== 단발 TV 시청시간 상위 3행 ===")
print(dtv.nlargest(3, "weekly_hours_viewed")[["week", "show_title", "st", "weekly_hours_viewed"]].to_string(index=False))
print("\n=== 단발 TV 시즌별 잔존(마지막 주 제외, 작품별 잔존행/전체행) ===")
d = sub[sub["grp3"] == "단발 TV"]
per = d.groupby(["show_title", "st"])["survives"].agg(["count", "sum"])
per["잔존주수"] = per["count"]
print(per.rename(columns={"sum": "다음주잔존행"}).to_string())
