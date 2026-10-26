# -*- coding: utf-8 -*-
"""1단계. 데이터 이해 — TL_2.소비자.zip 구조 점검"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd

SRC = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"

fin = pd.read_csv(SRC, encoding="utf-8-sig")
print("=== 전체 데이터 ===")
print("행 수:", f"{len(fin):,}", "/ 열 수:", len(fin.columns))
print()
print("=== 열 목록 (이름 / dtype / 결측 아님 값 개수 / 예시값 2개) ===")
for c in fin.columns:
    ex = fin[c].dropna().unique()[:2]
    print(f"{c:28s} | {str(fin[c].dtype):8s} | nunique={fin[c].nunique():>7,} | 예시: {list(ex)}")
print()
print("=== customer_type 분포 ===")
print(fin["customer_type"].value_counts())
