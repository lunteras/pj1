import pandas as pd

df = pd.read_csv(
    r'ABA_1st_proj_시작파일\[바이오] Diabetes Health Indicators Dataset\diabetes_binary_health_indicators_BRFSS2015.csv'
)

def age_group(a):
    if a >= 10: return '65세 이상'
    if a >= 6:  return '45-64세'
    return '18-44세'

df['AgeG'] = df['Age'].apply(age_group)
age_order = ['18-44세', '45-64세', '65세 이상']

strata = [
    (0, 25,  'BMI < 25 (정상)'),
    (25, 30, 'BMI 25-29.9 (과체중)'),
    (30, 100,'BMI 30 이상 (비만)'),
]

print('=== BMI 층별 (연령 x 고혈압) 당뇨 유병률 (%) ===')
for lo, hi, name in strata:
    sub = df[(df['BMI'] >= lo) & (df['BMI'] < hi)]
    print(f'--- {name}  (n={len(sub):,}, 유병률 {sub["Diabetes_binary"].mean()*100:.1f}%) ---')
    pt = sub.pivot_table(index='AgeG', columns='HighBP', values='Diabetes_binary', aggfunc='mean')*100
    pt = pt.reindex(age_order).round(1)
    pt.columns = ['고혈압X', '고혈압O']
    pt['격차(%p)'] = (pt['고혈압O'] - pt['고혈압X']).round(1)
    print(pt.to_string())
    print()

# BMI 자체 효과: 고혈압이 같은 그룹 안에서 BMI 유무로 나눠 비교
print('=== 고혈압 층별 (연령 x BMI 비만 여부) 당뇨 유병률 (%) ===')
for bp in [0, 1]:
    sub = df[df['HighBP'] == bp]
    label = '고혈압 있음' if bp == 1 else '고혈압 없음'
    print(f'--- {label} (n={len(sub):,}) ---')
    sub = sub.copy()
    sub['Obese'] = (sub['BMI'] >= 30).map({0: '비만X', 1: '비만O'})
    pt = sub.pivot_table(index='AgeG', columns='Obese', values='Diabetes_binary', aggfunc='mean')*100
    pt = pt.reindex(age_order).round(1)
    print(pt.to_string())
    print()
