import pandas as pd

df = pd.read_csv(
    r'ABA_1st_proj_시작파일\[바이오] Diabetes Health Indicators Dataset\diabetes_binary_health_indicators_BRFSS2015.csv'
)
age_map = {1:'18-24',2:'25-29',3:'30-34',4:'35-39',5:'40-44',6:'45-49',7:'50-54',
           8:'55-59',9:'60-64',10:'65-69',11:'70-74',12:'75-79',13:'80+'}
df['AgeG'] = df['Age'].map(age_map)

print('=== 전체 당뇨 유병률: {:.1f}% ==='.format(df['Diabetes_binary'].mean()*100))

print()
print('=== 단일 조건별 당뇨 유병률 ===')
for col in ['HighBP','HighChol']:
    for v in [0,1]:
        sub = df[df[col]==v]
        label = '보유' if v==1 else '미보유'
        print(f'{col} {label}: {sub["Diabetes_binary"].mean()*100:.1f}%  (n={len(sub):,})')

print()
print('=== 연령대별 당뇨 유병률 (%) ===')
g = (df.groupby('AgeG')['Diabetes_binary'].mean()*100).round(1)
print(g.to_string())

print()
print('=== 연령대 x 고혈압 보유 여부: 당뇨 유병률 (%) ===')
pt = df.pivot_table(index='AgeG', columns='HighBP', values='Diabetes_binary', aggfunc='mean')*100
pt = pt.round(1)
pt.columns = ['고혈압 없음', '고혈압 있음']
pt['격차(배수)'] = (pt['고혈압 있음'] / pt['고혈압 없음']).round(1)
print(pt.to_string())

print()
print('=== 연령대 x 고혈압 x 고콜레스테롤: 당뇨 유병률 (%) ===')
pt2 = df.pivot_table(index='AgeG', columns=['HighBP','HighChol'], values='Diabetes_binary', aggfunc='mean')*100
pt2 = pt2.round(1)
pt2.columns = ['고혈압X,콜레스테롤X','고혈압X,콜레스테롤O','고혈압O,콜레스테롤X','고혈압O,콜레스테롤O']
print(pt2.to_string())
