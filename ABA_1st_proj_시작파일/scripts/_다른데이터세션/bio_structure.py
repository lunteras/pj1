import glob, os, pandas as pd

base = r"C:\Users\user\Desktop\1차프로젝트 시작파일\ABA_1st_proj_시작파일"
# 재귀 탐색으로 대괄호 폴더명 문제 회피
matches = glob.glob(os.path.join(base, "**", "*.csv"), recursive=True)
bio_path = [m for m in matches if "Diabetes" in m][0]

print("파일:", os.path.basename(bio_path))
print("폴더:", os.path.basename(os.path.dirname(bio_path)))

bio = pd.read_csv(bio_path)
print("행,열:", bio.shape)
print("결측치 총합:", int(bio.isna().sum().sum()))
print("중복 행 수:", int(bio.duplicated().sum()))
print("파일 크기(MB):", round(os.path.getsize(bio_path) / 1024 / 1024, 1))
print()
print("타겟 분포 (0=정상, 1=당뇨):")
print(bio["Diabetes_binary"].value_counts().to_string())
print()
for c in bio.columns:
    u = bio[c].nunique()
    vals = sorted(bio[c].unique())
    if u <= 15:
        print(f"{c}: 고유값 {u}개 -> {vals}")
    else:
        print(f"{c}: 고유값 {u}개, 범위 {min(vals)}~{max(vals)}, 평균 {bio[c].mean():.1f}, 중앙값 {bio[c].median():.1f}")
