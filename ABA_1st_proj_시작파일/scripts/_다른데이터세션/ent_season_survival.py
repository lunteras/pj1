import sys
import re
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ent = pd.read_csv(r"[엔터] Netflix Top 10 Weekly Dataset/all-weeks-global.csv", parse_dates=["week"])
ent["st"] = ent["season_title"].fillna("")
ent["key"] = list(zip(ent["category"], ent["show_title"], ent["st"]))
pair_set = set(zip(ent["key"], ent["week"]))
ent["survives"] = [(k, w + pd.Timedelta(days=7)) in pair_set for k, w in zip(ent["key"], ent["week"])]

last = ent["week"].max()
print("마지막 주 행(다음 주 데이터가 없어 판단 불가):", (ent["week"] == last).sum())

f = ent[(~ent["category"].str.startswith("TV")) & (ent["st"] != "")]
print("\n=== season_title이 비어 있지 않은 영화 행 ===")
print(f[["week", "show_title", "st"]].to_string(index=False) if len(f) else "없음")

tv = ent[ent["category"].str.startswith("TV") & (ent["st"] != "")]
sfx_n = tv["st"].str.split(":", n=1).str[1].str.strip().fillna("(콜론 없음)").str.replace(r"\d+", "#", regex=True)
print("\n=== TV 시즌 표기 패턴 분포(행 수) ===")
print(sfx_n.value_counts().to_string())

def season_num(s):
    m = re.search(r"(?:Season|Part|Volume|Vol\.?)\s*(\d+)", s)
    return int(m.group(1)) if m else None

def report(name, is_season_mask):
    ent["grp"] = "비시즌제(영화·단발)"
    ent.loc[is_season_mask, "grp"] = "시즌제 TV"
    sub = ent[ent["week"] != last]
    tab = sub.groupby("grp")["survives"].agg(행_수="count", 잔존비율=lambda s: round(100 * s.mean(), 1))
    print(f"\n=== {name} ===")
    print(tab.to_string())
    n_show = sub.groupby(["grp", "category", "show_title", "st"], dropna=False).size().reset_index().groupby("grp").size()
    print("작품(시즌) 수:", n_show.to_dict())

tv_mask = ent["category"].str.startswith("TV")
report("정의 A: Season/Part/Volume 숫자 표기 TV",
       tv_mask & ent["st"].str.contains(r"(?:Season|Part|Volume|Vol\.?)\s*\d", regex=True))
report("정의 B: Season 숫자 표기 TV만",
       tv_mask & ent["st"].str.contains(r"Season\s*\d", regex=True))

# 참고용: 시즌 1 vs 시즌 2 이상
ent["snum"] = ent["st"].map(season_num)
ent["sgrp"] = "비시즌제"
ent.loc[tv_mask & ent["snum"].notna(), "sgrp"] = "시즌 1"
ent.loc[tv_mask & (ent["snum"] >= 2), "sgrp"] = "시즌 2 이상"
sub = ent[ent["week"] != last]
print("\n=== 참고: 시즌 1 vs 시즌 2 이상 ===")
print(sub.groupby("sgrp")["survives"].agg(행_수="count", 잔존비율=lambda s: round(100 * s.mean(), 1)).to_string())
