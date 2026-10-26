import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

path = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[유통] 서울시 골목상권분석정보\서울시 상권분석서비스(추정매출-상권)_2024년.zip"
ret = pd.read_csv(path, encoding="cp949")

cafe = ret[ret["서비스_업종_코드_명"] == "커피-음료"]

for q in sorted(cafe["기준_년분기_코드"].unique()):
    kt = cafe[(cafe["기준_년분기_코드"] == q) & (cafe["상권_구분_코드_명"] == "관광특구")]
    ds = cafe[(cafe["기준_년분기_코드"] == q) & (cafe["상권_구분_코드_명"] == "발달상권")]
    m_kt = kt["당월_매출_금액"].median() / 1e8
    m_ds = ds["당월_매출_금액"].median() / 1e8
    print(f"{q}분기: 관광특구 {m_kt:.1f}억 ({len(kt)}행) | 발달상권 {m_ds:.1f}억 ({len(ds)}행) | {m_kt/m_ds:.1f}배")