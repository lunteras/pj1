import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

fin = pd.read_csv(
    r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip",
    encoding="utf-8-sig",
)

# 투자성향은 증권 거래에만 있음 (보험은 '-')
sec = fin[fin["customer_type"] == "증권"].copy()
print("증권 행 수:", len(sec))

blank_inc = sec["income_bracket"].isin(["미수집", "미답변", "-", "기타"]).sum()
blank_ip = sec["investment_propensity"].isin(["미수집", "미답변", "-", "기타"]).sum()
print("거른 행: 소득구간 빈칸", blank_inc, "건 / 투자성향 빈칸", blank_ip, "건")

# 고객당 소득구간/투자성향이 여러 개인 고객이 있는지
p_var = sec.groupby("customer_id")["investment_propensity"].nunique()
i_var = sec.groupby("customer_id")["income_bracket"].nunique()
print("투자성향이 2개 이상인 고객:", (p_var > 1).sum(), "명 / 소득구간이 2개 이상인 고객:", (i_var > 1).sum(), "명")

order = ["1000만원미만", "3000만원미만", "5000만원미만", "8000만원미만",
         "10000만원미만", "10000만원이상", "30000만원미만"]

print()
print("=== 행 단위 (증권 거래 전체) ===")
ct = pd.crosstab(sec["income_bracket"], sec["investment_propensity"]).reindex(order)
pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)
tab = pd.DataFrame({"행 수": ct.sum(axis=1)})
tab["안정형(%)"] = pct["안정형"]
tab["안정추구형(%)"] = pct["안정추구형"]
tab["안정형+안정추구형(%)"] = (pct["안정형"] + pct["안정추구형"]).round(1)
tab["공격투자형(%)"] = pct["공격투자형"]
print(tab.to_string())
print()
print("5종 전체 분포(행 단위, %):")
print(pct.to_string())
print()

print("=== 고객 단위 (고객당 1행, 중복 제거) ===")
cust = sec.drop_duplicates(subset="customer_id")
print("고객 수:", len(cust))
ct2 = pd.crosstab(cust["income_bracket"], cust["investment_propensity"]).reindex(order)
pct2 = (ct2.div(ct2.sum(axis=1), axis=0) * 100).round(1)
tab2 = pd.DataFrame({"고객 수": ct2.sum(axis=1)})
tab2["안정형(%)"] = pct2["안정형"]
tab2["안정추구형(%)"] = pct2["안정추구형"]
tab2["안정형+안정추구형(%)"] = (pct2["안정형"] + pct2["안정추구형"]).round(1)
tab2["공격투자형(%)"] = pct2["공격투자형"]
print(tab2.to_string())