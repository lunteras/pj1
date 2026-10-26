import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

path = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[유통] 서울시 골목상권분석정보\서울시 상권분석서비스(추정매출-상권)_2024년.zip"
ret = pd.read_csv(path, encoding="cp949")

cafe = ret[ret["서비스_업종_코드_명"] == "커피-음료"]
sub = cafe[cafe["상권_구분_코드_명"].isin(["관광특구", "발달상권"])]

t = sub.groupby("상권_구분_코드_명")["당월_매출_금액"].agg(["count", "median", "mean"])
print(t.to_string())

print("\n--- 억원 환산 ---")
for name, row in t.iterrows():
    print(f"{name}: 행수={row['count']:,} 중앙값={row['median']/1e8:.1f}억 평균={row['mean']/1e8:.1f}억")

med_r = t.loc["관광특구", "median"]; med_d = t.loc["발달상권", "median"]
mean_r = t.loc["관광특구", "mean"]; mean_d = t.loc["발달상권", "mean"]
print(f"\n중앙값 배수: {med_r/med_d:.2f}배 | 평균 배수: {mean_r/mean_d:.2f}배")

print("\n--- 상권 수 (커피-음료 존재 상권) ---")
print(sub.groupby("상권_구분_코드_명")["상권_코드"].nunique().to_string())

print("\n--- 관광특구 커피-음료 행 전체 (참고) ---")
kt = cafe[cafe["상권_구분_코드_명"] == "관광특구"]
print(kt[["기준_년분기_코드", "상권_코드_명", "당월_매출_금액"]].sort_values(["상권_코드_명", "기준_년분기_코드"]).to_string(index=False))
