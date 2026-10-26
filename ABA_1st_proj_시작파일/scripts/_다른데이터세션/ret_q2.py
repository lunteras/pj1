import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from matplotlib import font_manager, pyplot as plt

path = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[유통] 서울시 골목상권분석정보\서울시 상권분석서비스(추정매출-상권)_2024년.zip"
ret = pd.read_csv(path, encoding="cp949")

cafe = ret[ret["서비스_업종_코드_명"] == "커피-음료"]
kt = cafe[cafe["상권_구분_코드_명"] == "관광특구"]
ds = cafe[cafe["상권_구분_코드_명"] == "발달상권"]

# --- 유별난 곳 판별용 숫자 ---
kt_no_md = kt[~kt["상권_코드_명"].str.startswith("명동")]
md = kt[kt["상권_코드_명"].str.startswith("명동")]
print(f"명동 분기 평균 매출: {md['당월_매출_금액'].mean()/1e8:.1f}억")
print(f"명동 제외 관광특구(5곳 20행) 분기 중앙값: {kt_no_md['당월_매출_금액'].median()/1e8:.1f}억, 평균: {kt_no_md['당월_매출_금액'].mean()/1e8:.1f}억")
print(f"명동 제외 관광특구 최소: {kt_no_md['당월_매출_금액'].min()/1e8:.1f}억 / 최대: {kt_no_md['당월_매출_금액'].max()/1e8:.1f}억")
print(f"발달상권 분기 중앙값: {ds['당월_매출_금액'].median()/1e8:.1f}억, 75%: {ds['당월_매출_금액'].quantile(0.75)/1e8:.1f}억, 90%: {ds['당월_매출_금액'].quantile(0.90)/1e8:.1f}억, 최대: {ds['당월_매출_금액'].max()/1e8:.1f}억")
kt_min = kt["당월_매출_금액"].min()
print(f"관광특구 최소 행(동대문 20241): {kt_min/1e8:.1f}억")
print(f"발달상권 중 관광특구 최소({kt_min/1e8:.1f}억)보다 큰 행 비율: {(ds['당월_매출_금액'] > kt_min).mean()*100:.1f}%")

# --- 상자 그림 ---
names = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot([kt["당월_매출_금액"], ds["당월_매출_금액"]], showmeans=True)
ax.set_xticks([1, 2])
ax.set_xticklabels(["관광특구 (6상권 24행)", "발달상권 (244상권 975행)"])
ax.set_yscale("log")
ax.set_title("커피-음료 상권 분기 매출 — 관광특구 vs 발달상권 (2024년)")
ax.set_ylabel("분기 매출 금액 (원, 로그 축)")
fig.tight_layout()
out = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\output\유통_커피매출_관광특구_발달상권_상자그림.png"
fig.savefig(out, dpi=150)
print("저장:", out)
