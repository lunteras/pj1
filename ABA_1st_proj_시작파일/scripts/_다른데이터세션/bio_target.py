import glob, os, pandas as pd

base = r"C:\Users\user\Desktop\1차프로젝트 시작파일\ABA_1st_proj_시작파일"
matches = glob.glob(os.path.join(base, "**", "*.csv"), recursive=True)
bio_path = [m for m in matches if "Diabetes" in m][0]
bio = pd.read_csv(bio_path)

t = bio["Diabetes_binary"]
base_rate = t.mean() * 100
print(f"전체 당뇨 비율: {base_rate:.1f}%")
print()

# 0/1 열: 1일 때 당뇨 비율 vs 0일 때 비율
print("=== 0/1 열 (값=1일 때 당뇨비율 vs 값=0일 때 당뇨비율) ===")
rows = []
for c in bio.columns:
    if c == "Diabetes_binary":
        continue
    if bio[c].nunique() == 2:
        r1 = bio.loc[bio[c] == 1, "Diabetes_binary"].mean() * 100
        r0 = bio.loc[bio[c] == 0, "Diabetes_binary"].mean() * 100
        rows.append((c, r0, r1, r1 - r0))
rows.sort(key=lambda x: -abs(x[3]))
for c, r0, r1, d in rows:
    print(f"{c:22s} 0일때 {r0:4.1f}% / 1일때 {r1:4.1f}%  (차이 {d:+.1f}%p)")

print()
print("=== 구간 열: 당뇨 비율이 구간에 따라 어떻게 변하는지 ===")
for c in ["GenHlth", "Age", "Income", "Education"]:
    rates = bio.groupby(c)["Diabetes_binary"].mean() * 100
    lo, hi = rates.iloc[0], rates.iloc[-1]
    print(f"{c}: {c}={rates.index[0]:.0f}일때 {lo:.1f}% -> {c}={rates.index[-1]:.0f}일때 {hi:.1f}%")

print()
print("=== BMI 4구간별 당뇨 비율 ===")
bio["BMI구간"] = pd.cut(bio["BMI"], [0, 18.5, 25, 30, 100],
                       labels=["저체중(~18.5)", "정상(18.5~25)", "과체중(25~30)", "비만(30~)"])
print((bio.groupby("BMI구간", observed=True)["Diabetes_binary"].mean() * 100).round(1).to_string())
