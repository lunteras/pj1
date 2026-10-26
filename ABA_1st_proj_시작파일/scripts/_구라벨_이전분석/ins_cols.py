import sys, re
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

fin = pd.read_csv("[금융] 금융상품·서비스 및 소비자 특성 데이터/Training/02.라벨링데이터/TL_2.소비자.zip", encoding="utf-8-sig")
print("전체 행, 열:", fin.shape)
print("열 목록:", list(fin.columns))
print()

id_cols = [c for c in fin.columns if re.search(r"(id|번호|일련|sn$|_sn|no$|_no)", str(c), re.I) or "고객" in str(c)]
print("고객 ID 후보 열:", id_cols)
for c in id_cols:
    print(f"  {c}: 고유값 {fin[c].nunique():,}개 / 전체 행 {len(fin):,}")
