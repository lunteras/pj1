# -*- coding: utf-8 -*-
"""1단계. 데이터 이해 — 고객 키 성격 확인"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd

SRC = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"
fin = pd.read_csv(SRC, encoding="utf-8-sig")

both = fin.groupby("customer_id")["customer_type"].nunique()
print("customer_id 총:", f"{fin['customer_id'].nunique():,}")
print("보험+증권 둘 다 가진 고객:", f"{(both>1).sum():,}")

ins = fin[fin["customer_type"]=="보험"]
multi = ins[ins.duplicated("customer_id", keep=False)].sort_values(["customer_id","customer_seq"])
print()
print("=== 보험 고객 중 거래 2건 이상인 고객의 행 구조 예시 (첫 고객 2명) ===")
ids = multi["customer_id"].unique()[:2]
cols = ["customer_id","customer_seq","cross_coverage","product_name","transaction_keyword",
        "transaction_amt","product_amt","product_period","transaction_type","transaction_start_year"]
print(multi[multi["customer_id"].isin(ids)][cols].to_string(index=False))

print()
print("=== customer_seq별 행 수 (보험) ===")
print(ins["customer_seq"].value_counts().sort_index())

print()
print("=== customer_id + customer_seq 조합의 유일성 ===")
print("중복 조합 수:", ins.duplicated(["customer_id","customer_seq"]).sum())
print("중복 (customer_id, product_name) 수:", ins.duplicated(["customer_id","product_name"]).sum())
