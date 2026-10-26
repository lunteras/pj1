import pandas as pd

path = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[유통] 서울시 골목상권분석정보\서울시 상권분석서비스(추정매출-상권)_2024년.zip"
ret = pd.read_csv(path, encoding="cp949")

print("=== shape ===")
print(ret.shape)

print("\n=== columns & dtypes ===")
for c in ret.columns:
    nn = ret[c].notna().sum()
    nu = ret[c].nunique()
    sample = ret[c].dropna().unique()[:3]
    print(f"{c} | dtype={ret[c].dtype} | 결측={len(ret)-nn} | 고유값수={nu} | 예시={list(sample)}")

print("\n=== 상권_구분_코드_명 별 행 수 ===")
print(ret["상권_구분_코드_명"].value_counts())

print("\n=== 기준_년분기_코드 별 행 수 ===")
print(ret["기준_년분기_코드"].value_counts().sort_index())

print("\n=== 커피-음료만 ===")
cafe = ret[ret["서비스_업종_코드_명"] == "커피-음료"]
print("행 수:", len(cafe))
print("상권 수:", cafe["상권_코드"].nunique() if "상권_코드" in cafe.columns else "?")
print(cafe["상권_구분_코드_명"].value_counts())

print("\n=== 업종 고유값 (커피 관련만) ===")
print([u for u in ret["서비스_업종_코드_명"].unique() if "커피" in str(u) or "음료" in str(u) or "카페" in str(u)])

print("\n=== 당월_매출_금액 요약 (커피-음료) ===")
print(cafe["당월_매출_금액"].describe())
