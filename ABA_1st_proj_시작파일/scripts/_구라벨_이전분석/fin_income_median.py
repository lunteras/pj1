import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

fin = pd.read_csv(
    r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip",
    encoding="utf-8-sig",
)

sec = fin[fin["customer_type"] == "증권"]
cust = sec.drop_duplicates(subset="customer_id")

order = ["1000만원미만", "3000만원미만", "5000만원미만", "8000만원미만",
         "10000만원미만", "10000만원이상", "30000만원미만"]

vc = cust["income_bracket"].value_counts().reindex(order).dropna().astype(int)
n = len(cust)
tab = pd.DataFrame({
    "고객 수": vc.values,
    "비율(%)": (vc.values / n * 100).round(1),
    "누적 비율(%)": (vc.values.cumsum() / n * 100).round(1),
}, index=vc.index)
print("고객 수 합계:", n)
print(tab.to_string())
print()
half = n / 2
cum = vc.values.cumsum()
idx = int((cum >= half).argmax())
print(f"중앙 위치: {half:,.1f}번째 고객 -> {vc.index[idx]} 구간 (누적 {cum[idx]:,}명에서 처음 절반 넘음)")
print("최빈 구간:", vc.idxmax(), f"({vc.max():,}명, {vc.max()/n*100:.1f}%)")