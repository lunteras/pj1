import sys, re
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

fin = pd.read_csv("[금융] 금융상품·서비스 및 소비자 특성 데이터/Training/02.라벨링데이터/TL_2.소비자.zip", encoding="utf-8-sig")
ins = fin[fin["customer_type"] == "보험"].copy()
blanks = ["미수집", "미답변", "-", "기타"]

sub = ins[~(ins["credit_rating"].isin(blanks) | ins["credit_rating"].isna())].copy()
sub["transaction_amt"] = pd.to_numeric(sub["transaction_amt"], errors="coerce")
sub = sub.dropna(subset=["transaction_amt"])

cust = sub.groupby("customer_id").agg(total_amt=("transaction_amt", "sum"), band=("credit_rating", "first"), income=("income_bracket", "first"))
print("고객 수:", f"{len(cust):,}")
print()
print("income_bracket 값 분포 (고객 단위):")
print(cust["income"].value_counts(dropna=False).to_string())
print()

inc_blank = cust["income"].isin(blanks) | cust["income"].isna()
print("소득 빈값으로 뺀 고객 수:", f"{int(inc_blank.sum()):,}")
c = cust[~inc_blank].copy()

order = ["1000만원미만", "3000만원미만", "5000만원미만", "8000만원미만", "10000만원미만", "10000만원이상", "30000만원미만"]
order = [o for o in order if o in set(c["income"])]

print()
print("=== 소득 구간별 고객당 총 납입액 ===")
stats = c.groupby("income")["total_amt"].agg(["count", "median", "mean"]).reindex(order)
for o in order:
    r = stats.loc[o]
    print(f"{o:<12} | 고객 {int(r['count']):>7,}명 | 중앙값 {r['median']/1e4:,.1f}만원 | 평균 {r['mean']/1e4:,.1f}만원")
med = stats["median"]
print(f"중앙값 최고: {med.idxmax()} {med.max()/1e4:.1f}만원 / 최저: {med.idxmin()} {med.min()/1e4:.1f}만원 → {med.max()/med.min():.1f}배")
print("고객 수 30 미만 구간:", [o for o in order if int(stats.loc[o, "count"]) < 30] or "없음")
print()

def band_key(s):
    m = re.search(r"(\d{3})", str(s))
    return int(m.group(1)) if m else -1

bands = sorted(c["band"].unique(), key=band_key, reverse=True)
print("=== 소득 구간 × 신용점수대: 평균 총 납입액 (만원) / 괄호는 고객 수 ===")
pivot = c.pivot_table(index="income", columns="band", values="total_amt", aggfunc="mean").reindex(order)[bands]
cnt = c.pivot_table(index="income", columns="band", values="total_amt", aggfunc="count").reindex(order)[bands]
header = "소득 구간".ljust(12) + " | " + " | ".join(f"{b:>12}" for b in bands)
print(header)
for o in order:
    cells = []
    for b in bands:
        m = pivot.loc[o, b]
        n = cnt.loc[o, b]
        cells.append(f"{m/1e4:>6.1f} ({int(n):>5,})" if pd.notna(m) else f"{'-':>12}")
    print(o.ljust(12) + " | " + " | ".join(f"{c_:>12}" for c_ in cells))
print()
small_cells = [(o, b, int(cnt.loc[o, b])) for o in order for b in bands if pd.notna(cnt.loc[o, b]) and int(cnt.loc[o, b]) < 30]
print("고객 수 30 미만 칸:", small_cells if small_cells else "없음")
