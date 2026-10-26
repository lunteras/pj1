# -*- coding: utf-8 -*-
"""1단계. 데이터 이해 — 데이터 품질 점검 (대체값, 중복, 범주 값 목록)"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd

SRC = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"
fin = pd.read_csv(SRC, encoding="utf-8-sig")
ins = fin[fin["customer_type"] == "보험"].copy()
cust = ins.drop_duplicates("customer_id")  # 속성 열은 고객 내 고정 확인됨

print("=== 완전 중복 행 (보험 거래 77,305건 중) ===")
print("중복 행 수:", ins.duplicated().sum())

print()
print("=== 대체값(-, 미수집, 미답변, 기타) 분포 — 고객 단위 74,776명 ===")
for c in ["investment_propensity","disease_history","occupation_group","region",
          "credit_rating","risk_grade","marital_status","gender","age","income_bracket"]:
    vc = cust[c].value_counts(dropna=False)
    subst = {k: v for k, v in vc.items() if str(k).strip() in ["-","미수집","미답변","기타","nan"]}
    print(f"\n[{c}] 고유값 {cust[c].nunique()}개")
    print(vc.to_string())

print()
print("=== disease_history '기타' — 보장only vs 다중 담보(보장 포함) 고객 수 ===")
multi = ["보장+변액","보장+기타","보장+변액+기타"]
g = cust["cross_coverage"].isin(multi)
print("보장only 전체:", (~g).sum(), "중 기타:", ((~g) & (cust["disease_history"]=="기타")).sum())
print("다중 담보 전체:", g.sum(), "중 기타:", (g & (cust["disease_history"]=="기타")).sum())

print()
print("=== product_amt / transaction_amt / product_period 요약 (보험 거래) ===")
print(ins[["product_amt","transaction_amt","product_period"]].describe().to_string())
print("product_period 예시연도 해석용: 최소", ins["product_period"].min(), "/ 최대", ins["product_period"].max())
print("transaction_last_year 최대:", ins["transaction_last_year"].max())
