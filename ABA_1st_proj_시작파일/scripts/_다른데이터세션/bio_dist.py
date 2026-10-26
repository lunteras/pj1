import pandas as pd, json

df = pd.read_csv(
    r'ABA_1st_proj_시작파일\[바이오] Diabetes Health Indicators Dataset\diabetes_binary_health_indicators_BRFSS2015.csv'
)

out = {}

# 연령 라벨
age_labels = ['18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80+']

for grp in [0, 1]:
    d = df[df['Diabetes_binary'] == grp]
    key = 'no_dm' if grp == 0 else 'dm'

    # 연령 히스토그램 (13 bins)
    age_hist = d['Age'].value_counts().reindex(range(1, 14), fill_value=0).tolist()

    # BMI 히스토그램 (bin width 5, 12~98)
    bins = list(range(10, 105, 5))
    bmi_cut = pd.cut(d['BMI'], bins=bins)
    bmi_hist = (bmi_cut.value_counts().sort_index().tolist())
    bmi_labels = [f'{b}-{b+4}' for b in bins[:-1]]

    # PhysHlth (0-30) 히스토그램
    ph_hist = d['PhysHlth'].value_counts().reindex(range(0, 31), fill_value=0).tolist()

    out[key] = {
        'n': int(len(d)),
        'age_hist': age_hist,
        'age_mean': round(float(d['Age'].mean()), 2),
        'age_median': float(d['Age'].median()),
        'bmi_labels': bmi_labels,
        'bmi_hist': bmi_hist,
        'bmi_mean': round(float(d['BMI'].mean()), 2),
        'bmi_median': float(d['BMI'].median()),
        'ph_hist': ph_hist,
        'ph_mean': round(float(d['PhysHlth'].mean()), 2),
        'ph_median': float(d['PhysHlth'].median()),
    }

# 당뇨율도 같이
out['rate_overall'] = round(df['Diabetes_binary'].mean()*100, 1)

with open(r'C:\Users\user\AppData\Local\Temp\helpycode\bio_dist.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False)

# 콘솔 요약
for key, label in [('no_dm','당뇨 없음'), ('dm','당뇨 있음')]:
    d = out[key]
    print(f"[{label}] n={d['n']:,}")
    print(f"  Age   : mean={d['age_mean']}, median={d['age_median']}")
    print(f"  BMI   : mean={d['bmi_mean']}, median={d['bmi_median']}")
    print(f"  PhysHlth: mean={d['ph_mean']}, median={d['ph_median']}")
print('saved json ok')
