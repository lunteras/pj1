# 1차 프로젝트 시작 파일

4개 산업에서 **처음 여는 파일 하나씩**만 담았습니다. 전체 데이터는 `ABA_1st_proj.zip`에 있고, 파일 구성과 주의할 점은 그 안의 README에 정리되어 있습니다. 폴더 이름과 경로는 전체 데이터와 같아서, 여기서 쓴 코드는 전체 데이터에서도 그대로 돌아갑니다.

이 폴더를 그대로 열고 헬피코드를 실행하면 됩니다. `AGENTS.md`는 헬피코드가 읽는 규칙 파일이라 열어 보지 않아도 됩니다.

```python
import pandas as pd

# 바이오. 설문 응답자 1명이 1행, 253,680행
bio = pd.read_csv("[바이오] Diabetes Health Indicators Dataset/diabetes_binary_health_indicators_BRFSS2015.csv")

# 엔터. 작품 하나의 한 주가 1행, 1,160행
ent = pd.read_csv("[엔터] Netflix Top 10 Weekly Dataset/all-weeks-global.csv", parse_dates=["week"])

# 금융. 거래 1건이 1행, 385,019행
fin = pd.read_csv("[금융] 금융상품·서비스 및 소비자 특성 데이터/Training/02.라벨링데이터/TL_2.소비자.zip", encoding="utf-8-sig")

# 유통. 상권 하나의 업종 하나의 분기 하나가 1행, 2024년 4개 분기 87,179행
ret = pd.read_csv("[유통] 서울시 골목상권분석정보/서울시 상권분석서비스(추정매출-상권)_2024년.zip", encoding="cp949")
```

금융과 유통은 `encoding`을 빼면 글자가 깨지거나 오류가 납니다. 압축 파일은 풀지 않고 그대로 읽힙니다.
