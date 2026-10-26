import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

path = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[유통] 서울시 골목상권분석정보\서울시 상권분석서비스(추정매출-상권)_2024년.zip"
ret = pd.read_csv(path, encoding="cp949")

cafe = ret[ret["서비스_업종_코드_명"] == "커피-음료"]
kt = cafe[cafe["상권_구분_코드_명"] == "관광특구"]
ds = cafe[cafe["상권_구분_코드_명"] == "발달상권"]

print("=== 건수 0 행 확인 ===")
for name, g in [("관광특구", kt), ("발달상권", ds)]:
    zero = (g["당월_매출_건수"] == 0).sum()
    print(f"{name}: 매출건수 0인 행 {zero}개 (전체 {len(g)}행)")

print("\n=== 건당 매출 (원) ===")
for name, g in [("관광특구", kt), ("발달상권", ds)]:
    valid = g[g["당월_매출_건수"] > 0]
    per = valid["당월_매출_금액"] / valid["당월_매출_건수"]
    print(f"{name}: 행수={len(valid):,} 중앙값={per.median():,.0f}원 평균={per.mean():,.0f}원")

print("\n=== 주말 매출 비중 (참고용 계산만) ===")
for name, g in [("관광특구", kt), ("발달상권", ds)]:
    share = g["주말_매출_금액"] / g["당월_매출_금액"]
    print(f"{name}: 중앙값={share.median()*100:.1f}% 평균={share.mean()*100:.1f}%")
