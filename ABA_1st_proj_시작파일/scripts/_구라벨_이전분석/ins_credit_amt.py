import sys, re
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

fin = pd.read_csv("[금융] 금융상품·서비스 및 소비자 특성 데이터/Training/02.라벨링데이터/TL_2.소비자.zip", encoding="utf-8-sig")
print("전체 행, 열:", fin.shape)
print()
print("customer_type 분포:")
print(fin["customer_type"].value_counts(dropna=False).to_string())
print()

ins = fin[fin["customer_type"] == "보험"].copy()
print("보험 거래 행 수:", f"{len(ins):,}")
print()
print("credit_rating 분포 (보험 거래):")
print(ins["credit_rating"].value_counts(dropna=False).to_string())
print()

blanks = ["미수집", "미답변", "-", "기타"]
is_blank = ins["credit_rating"].isin(blanks) | ins["credit_rating"].isna()
print("빈값(미수집·미답변·-·기타·결측)으로 뺀 행 수:", f"{int(is_blank.sum()):,}")
print()

sub = ins[~is_blank].copy()
if sub["transaction_amt"].dtype == object:
    sub["transaction_amt"] = pd.to_numeric(
        sub["transaction_amt"].astype(str).str.replace(",", ""), errors="coerce"
    )
    print("금액 열 문자열 → 숫자 변환, 변환 실패(결측) 행:", int(sub["transaction_amt"].isna().sum()))
print("보험 거래 transaction_amt 통계:")
print(sub["transaction_amt"].describe().to_string())
print()

def band_key(s):
    m = re.search(r"(\d{3})", str(s))
    return int(m.group(1)) if m else -1

bands = sorted(sub["credit_rating"].unique(), key=band_key, reverse=True)
print("집단 순서:", bands)
print()

g = sub.groupby("credit_rating")["transaction_amt"]
tbl = g.agg(["count", "median", "mean"]).reindex(bands)
for b in bands:
    c, med, mean = tbl.loc[b, "count"], tbl.loc[b, "median"], tbl.loc[b, "mean"]
    print(f"{b} | 행 수 {int(c):,} | 중앙값 {med:,.0f}원 ({med/1e4:,.1f}만원) | 평균 {mean:,.0f}원 ({mean/1e4:,.1f}만원)")
print()

med = tbl["median"]
print(f"중앙값 최고: {med.idxmax()} {med.max():,.0f}원 / 최저: {med.idxmin()} {med.min():,.0f}원 → {med.max()/med.min():.1f}배")
mean = tbl["mean"]
print(f"평균 최고: {mean.idxmax()} {mean.max():,.0f}원 / 최저: {mean.idxmin()} {mean.min():,.0f}원 → {mean.max()/mean.min():.1f}배")
print()
small = [b for b in bands if int(tbl.loc[b, "count"]) < 30]
print("행 수 30 미만 집단:", small if small else "없음")
