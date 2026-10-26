import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

path = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[유통] 서울시 골목상권분석정보\서울시 상권분석서비스(추정매출-상권)_2024년.zip"
ret = pd.read_csv(path, encoding="cp949")

cafe = ret[ret["서비스_업종_코드_명"] == "커피-음료"]
kt = cafe[cafe["상권_구분_코드_명"] == "관광특구"]
ds = cafe[cafe["상권_구분_코드_명"] == "발달상권"]

for name, g in [("관광특구", kt), ("발달상권", ds)]:
    jc, jm = g["주중_매출_금액"], g["주말_매출_금액"]
    share = jm / g["당월_매출_금액"]
    print(f"{name} (행수={len(g):,})")
    print(f"  주중 중앙값 {jc.median()/1e8:.1f}억 | 주중 평균 {jc.mean()/1e8:.1f}억")
    print(f"  주말 중앙값 {jm.median()/1e8:.1f}억 | 주말 평균 {jm.mean()/1e8:.1f}억")
    print(f"  주말 비중 중앙값 {share.median()*100:.1f}% | 평균 {share.mean()*100:.1f}%")