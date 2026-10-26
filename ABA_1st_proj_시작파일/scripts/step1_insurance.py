# -*- coding: utf-8 -*-
"""1단계. 데이터 이해 — 보험 고객 단위 집합 구조 검증"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd

SRC = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"
fin = pd.read_csv(SRC, encoding="utf-8-sig")
# 보험 거래만 — 증권 고객은 별도 부류(0명 겹침, 확인済)
ins = fin[fin["customer_type"] == "보험"].copy()

print("=== 보험 거래 ===")
print("거래(행) 수:", f"{len(ins):,}")
print("고객(customer_id) 수:", f"{ins['customer_id'].nunique():,}")
print("customer_seq 값:", sorted(ins["customer_seq"].unique()))

# 고객당 거래 수 분포
per = ins.groupby("customer_id").size()
print("고객당 거래 수: 중앙값", per.median(), "/ 최대", per.max(), "/ 1건인 고객", f"{(per==1).sum():,}명")

# 속성 열이 고객 내에서 고정인지 확인
attrs = ["age","gender","region","income_bracket","occupation_group","risk_grade",
         "credit_rating","marital_status","investment_propensity","cross_coverage",
         "disease_history","customer_type"]
varying = [a for a in attrs if (ins.groupby("customer_id")[a].nunique() > 1).any()]
print("고객 내에서 값이 변하는 속성 열:", varying if varying else "없음 (모두 고정)")

print()
print("=== cross_coverage 분포 (거래 건 기준) ===")
print(ins["cross_coverage"].value_counts())

# 고객 단위로 묶었을 때 (고객당 cross_coverage도 고정인지 → varying에 없으면 아래는 동일)
print()
print("=== cross_coverage 분포 (고객 수 기준) ===")
cust = ins.drop_duplicates("customer_id")
print(cust["cross_coverage"].value_counts())
print()
print("보험 고객 총:", f"{len(cust):,}")

print()
print("=== 보험 거래의 가입 연도 ===")
print(ins["transaction_start_year"].value_counts().sort_index())

print()
print("=== 보험 상품 종류 (product_name) 상위 15 ===")
print(ins["product_name"].value_counts().head(15))
