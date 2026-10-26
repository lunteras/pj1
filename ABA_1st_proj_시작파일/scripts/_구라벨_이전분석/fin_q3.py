import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

fin = pd.read_csv(
    r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip",
    encoding="utf-8-sig",
)

sec = fin[fin["customer_type"] == "증권"]
cust = sec.drop_duplicates(subset="customer_id")
print("증권 고객 수:", len(cust))
print("나이 라벨 종류:", cust["age"].value_counts().to_dict())

# 1) 나이대별 투자성향 비율
age_order = [a for a in ["20대", "30대", "40대", "50대", "60대", "70대", "70대 이상", "80대 이상"] if a in set(cust["age"])]
ct = pd.crosstab(cust["age"], cust["investment_propensity"]).reindex(age_order)
pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)
tab = pd.DataFrame({"고객 수": ct.sum(axis=1)})
for c in ["안정형", "안정추구형", "공격투자형"]:
    tab[c + "(%)"] = pct[c]
tab["안정형+안정추구형(%)"] = (pct["안정형"] + pct["안정추구형"]).round(1)
print()
print("=== 나이대별 투자성향 비율 (고객 단위) ===")
print(tab.to_string())

# 2) 소득 구간별 나이 구성 (50대 이상 비중) — 나이가 소득에 딸려 오는지
cust2 = cust.copy()
cust2["50대이상"] = cust2["age"].isin(["50대", "60대", "70대", "70대 이상", "80대 이상"])
inc_order = ["1000만원미만", "3000만원미만", "5000만원미만", "10000만원미만", "10000만원이상", "30000만원미만"]
g = cust2.groupby("income_bracket")["50대이상"].agg(["mean", "count"]).reindex(inc_order)
g["mean"] = (g["mean"] * 100).round(1)
g = g.rename(columns={"mean": "50대이상비중(%)"})
print()
print("=== 소득 구간별 50대 이상 고객 비중 ===")
print(g.to_string())

# 3) 나이를 통제한 확인: 40대 안에서 소득 구간별 공격투자형/안정형 비율
sub = cust[cust["age"] == "40대"]
ct3 = pd.crosstab(sub["income_bracket"], sub["investment_propensity"]).reindex(inc_order)
pct3 = (ct3.div(ct3.sum(axis=1), axis=0) * 100).round(1)
tab3 = pd.DataFrame({"고객 수": ct3.sum(axis=1)})
tab3["안정형(%)"] = pct3["안정형"]
tab3["공격투자형(%)"] = pct3["공격투자형"]
print()
print("=== 40대 안에서 소득 구간별 성향 (나이 통제) ===")
print(tab3.to_string())

# 4) 참고: 직업군 빈칸 규모 (이유 2 관련)
print()
print("증권 고객 occupation_group 미답변 비율:", round((cust["occupation_group"] == "미답변").mean() * 100, 1), "%")