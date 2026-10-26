import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

fin = pd.read_csv(
    r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip",
    encoding="utf-8-sig",
)

ok = fin[fin["product_period"] != 9999]
print("product_period >= transaction_start_year 비율(9999 제외):",
      round((ok["product_period"] >= ok["transaction_start_year"]).mean() * 100, 1), "%")
print()

sec1 = fin[(fin["customer_type"] == "증권") & (fin["customer_seq"] == 1)]
print("첫 거래(증권, seq=1) 행 수:", len(sec1), "/ 고유 고객:", sec1["customer_id"].nunique())
print()
print("첫 거래 상품 분포:")
print(sec1["transaction_keyword"].value_counts().to_string())
print()
print("첫 거래 투자성향 분포:")
print(sec1["investment_propensity"].value_counts().to_string())
print()
print("첫 거래 거래금액 중앙값:", f"{sec1['transaction_amt'].median():,.0f}", "원")
print("첫 거래 거래금액 평균:", f"{sec1['transaction_amt'].mean():,.0f}", "원")
print()
print("첫 거래 연령 분포:")
print(sec1["age"].value_counts().to_string())
