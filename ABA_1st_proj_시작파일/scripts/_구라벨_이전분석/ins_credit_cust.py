import sys, re
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

fin = pd.read_csv("[금융] 금융상품·서비스 및 소비자 특성 데이터/Training/02.라벨링데이터/TL_2.소비자.zip", encoding="utf-8-sig")
ins = fin[fin["customer_type"] == "보험"].copy()
blanks = ["미수집", "미답변", "-", "기타"]
sub = ins[~(ins["credit_rating"].isin(blanks) | ins["credit_rating"].isna())].copy()
sub["transaction_amt"] = pd.to_numeric(sub["transaction_amt"], errors="coerce")
sub = sub.dropna(subset=["transaction_amt"])

print("보험 거래(빈값 제외) 행 수:", f"{len(sub):,}")
print("보험 거래 고객 수:", f"{sub['customer_id'].nunique():,}")
print()

multi = sub.groupby("customer_id")["credit_rating"].nunique()
print("고객당 점수대가 2개 이상인 경우:", f"{int((multi > 1).sum()):,}명")
print()

def band_key(s):
    m = re.search(r"(\d{3})", str(s))
    return int(m.group(1)) if m else -1

cust = sub.groupby("customer_id").agg(
    total_amt=("transaction_amt", "sum"),
    n_tx=("transaction_amt", "count"),
    band=("credit_rating", "first"),
)
cust["band"] = pd.Categorical(cust["band"], ordered=True)
bands = sorted(cust["band"].unique(), key=band_key, reverse=True)

print(f"{'집단':<8} | 고객 수 | 총액 중앙값 | 총액 평균 | 평균 거래 건수")
med_total = cust.groupby("band", observed=True)["total_amt"]
stats = cust.groupby("band", observed=True).agg(
    n=("total_amt", "count"),
    med=("total_amt", "median"),
    mean=("total_amt", "mean"),
    tx=("n_tx", "mean"),
).reindex(bands)
for b in bands:
    r = stats.loc[b]
    print(f"{b:<8} | {int(r['n']):>7,} | {r['med']:>9,.0f}원 ({r['med']/1e4:.1f}만원) | {r['mean']:>9,.0f}원 ({r['mean']/1e4:.1f}만원) | {r['tx']:.2f}건")
print()

med = stats["med"]
print(f"총액 중앙값 최고: {med.idxmax()} {med.max():,.0f}원 / 최저: {med.idxmin()} {med.min():,.0f}원 → {med.max()/med.min():.1f}배")
mean = stats["mean"]
print(f"총액 평균 최고: {mean.idxmax()} {mean.max():,.0f}원 / 최저: {mean.idxmin()} {mean.min():,.0f}원 → {mean.max()/mean.min():.1f}배")
print()
print("고객 수 30 미만 집단:", [b for b in bands if int(stats.loc[b, "n"]) < 30] or "없음")
print()
print("고객당 거래 건수 분포 (전체):")
print(cust["n_tx"].describe().to_string())
