# -*- coding: utf-8 -*-
"""상담 대상 순위 리스트·보드 생성 — T=2024년 말 풀, v3 전체 풀 스코어 (2026-10-16)

[용도]
    보험 고객 풀에서 2025년(T 이후) 신규 가입 가능성을 v3 로지스틱 회귀로 점수화하고,
    운용 규칙(예외 후순위 3종)을 반영한 상담 대상 순위(CSV 2종)와 보드(HTML)를 만든다.

[프레임]
    데이터 : TL_2*.zip (작업 폴더 하위 os.walk 검색, utf-8-sig)
    대상   : customer_type=='보험', T=2024
    풀     : transaction_start_year<=2024 상품을 1개 이상 보유한 고객
             (customer_id 단위 drop_duplicates, 예상 38,932명)
    타깃 y : transaction_start_year>2024 상품 1개 이상 가입 = 1 (예상 1,137명, 2.9%)
    스코어 : 전체 풀로 학습 후 전 고객 점수(가입 확률) — in-sample이므로 검증·운용 참고용

[피처 v3 — 13계열]
    범주 8 : age, gender, region, income_bracket, occupation_group,
             risk_grade, credit_rating, marital_status → get_dummies(drop_first=True)
    수치 5 : tenure(=2024-고객별 최초가입연도), log1p(거래금액 transaction_amt 합),
             log1p(상품금액 product_amt 합), cnt2(과거 보유상품수>=2),
             rider(과거행 transaction_keyword에 '주보험'과 '특약' 동시 포함)
    교차   : age 원-핫 더미 × 기혼 지시자

[모델]
    전 피처 StandardScaler → LogisticRegression(max_iter=2000,
    class_weight='balanced', random_state=42) → predict_proba로 score 산출

[운용 규칙]
    예외 3종을 스코어 순위 뒤로 밀기
      ① region in ['대구','대전','인천']        → 저율지역 후순위
      ② age=='20대' and marital_status=='기혼'  → 20대 기혼 후순위
      ③ credit_rating=='300점대'                → 신용300 최후순위(맨 마지막)
    운용순위 : 예외 없는 고객을 score 내림차순 1~N 배치 → ①② 예외를 score순으로
               그 뒤에 → 신용300(③)을 맨 마지막에 score순 배치.
               예외가 겹치면 사유를 모두 표기하고, 신용300이 있으면 최후순위 그룹에 포함.
    운용티어 : 운용순위 <=500 'A', <=1000 'B', <=2000 'C', 그 외 '-'

[산출물]
    output/상담대상_순위_T2024.csv      : 전체 풀 순위표, utf-8-sig
    output/상담대상_TOP50_T2024.csv     : 상위 50행, 동일 열
    output/상담대상_순위보드_T2024.html : 자체 포함 HTML 보드 (외부 리소스 0)
    재현 : python scripts/step4_rank_list.py
"""
import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일"
OUT = os.path.join(BASE, "output")
T = 2024
LOW_REGIONS = ["대구", "대전", "인천"]
INC_BELOW5000 = ["1000만원미만", "3000만원미만", "5000만원미만"]

# ── 1. TL_2*.zip 검색(os.walk) 후 로드 ──────────────────────────
hits = []
for root, dirs, files in os.walk(BASE):
    for f in files:
        if f.startswith("TL_2") and f.endswith(".zip"):
            hits.append(os.path.join(root, f))
hits.sort()
print(f"[검색] TL_2*.zip {len(hits)}개 → {os.path.basename(hits[0])}")
fin = pd.read_csv(hits[0], encoding="utf-8-sig")
print(f"[데이터] {len(fin):,}행 × {fin.shape[1]}열")

ins = fin[fin["customer_type"] == "보험"].copy()
past = ins[ins["transaction_start_year"] <= T].copy()
fut = ins[ins["transaction_start_year"] > T].copy()

# ── 2. 풀과 타깃 ────────────────────────────────────────────────
# 풀: T 이전 상품을 1개 이상 보유한 고객 (고객 1행, 대표 행은 ins 첫 행)
pool = ins.drop_duplicates("customer_id").copy()
pool = pool[pool["customer_id"].isin(set(past["customer_id"]))].reset_index(drop=True)
y = pool["customer_id"].isin(set(fut["customer_id"])).astype(int).values
n = len(pool)
pos = int(y.sum())
print(f"[풀] {n:,}명 / 양성 {pos:,}명 ({pos/n*100:.2f}%)  (기대: 38,932 / 1,137 / 2.9%)")

# ── 3. 피처 v3 (모두 T 이전 행만 사용) ──────────────────────────
first_year = past.groupby("customer_id")["transaction_start_year"].min()
cnt = past.groupby("customer_id").size()
kw = past.groupby("customer_id")["transaction_keyword"].agg(
    lambda s: " ".join(s.fillna("").astype(str)))
rid = kw.str.contains("주보험") & kw.str.contains("특약")

pool["tenure"] = (T - pool["customer_id"].map(first_year)).astype(float)
pool["cnt2"] = pool["customer_id"].map(cnt).ge(2).astype(float)
pool["log_pay"] = np.log1p(pool["customer_id"].map(
    past.groupby("customer_id")["transaction_amt"].sum()).astype(float))
pool["log_amt"] = np.log1p(pool["customer_id"].map(
    past.groupby("customer_id")["product_amt"].sum()).astype(float))
pool["rider"] = pool["customer_id"].map(rid).fillna(False).astype(float)

CATS = ["age", "gender", "region", "income_bracket", "occupation_group",
        "risk_grade", "credit_rating", "marital_status"]
NUMS = ["tenure", "log_pay", "log_amt", "cnt2", "rider"]

Xcat = pd.get_dummies(pool[CATS], drop_first=True)
age_d = pd.get_dummies(pool["age"], drop_first=True)
cross = age_d.mul(pool["marital_status"].eq("기혼").astype(float), axis=0)
cross.columns = [f"ageXmar_{c}" for c in cross.columns]
X = pd.concat([Xcat, pool[NUMS].astype(float), cross.astype(float)], axis=1).astype(float)
print(f"[피처] 열 {X.shape[1]}개 = 범주 원핫 {Xcat.shape[1]} + 수치 5 + 교차 {len(cross.columns)}")

# ── 4. 전체 풀 학습 → 전 고객 스코어 ────────────────────────────
scaler = StandardScaler().fit(X)
clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
clf.fit(scaler.transform(X), y)
pool["score"] = clf.predict_proba(scaler.transform(X))[:, 1]
print("[모델] LogisticRegression(max_iter=2000, balanced, seed=42) 전체 풀 학습 완료")

# ── 5. 운용 규칙: 예외 3종 → 운용순위 → 운용티어 ────────────────
ex_region = pool["region"].isin(LOW_REGIONS)
ex_m20 = pool["age"].eq("20대") & pool["marital_status"].eq("기혼")
ex_c300 = pool["credit_rating"].eq("300점대")
print(f"[예외값 확인] 지역(대구·대전·인천) {int(ex_region.sum()):,}명 / "
      f"20대&기혼 {int(ex_m20.sum()):,}명 / 신용300점대 {int(ex_c300.sum()):,}명")

def flag_row(a, b, c):
    fs = []
    if a: fs.append("저율지역 후순위")
    if b: fs.append("20대 기혼 후순위")
    if c: fs.append("신용300 최후순위")
    return " · ".join(fs)

pool["예외사유"] = [flag_row(a, b, c) for a, b, c in zip(ex_region, ex_m20, ex_c300)]
# 그룹 0=예외 없음, 1=일반 예외(①②), 2=신용300(③, 최후순위 — 겹치면 ③ 우선)
pool["_grp"] = np.where(ex_c300, 2, np.where(ex_region | ex_m20, 1, 0))

ord_idx = pd.DataFrame({"s": pool["score"], "g": pool["_grp"]}) \
    .sort_values(["g", "s"], ascending=[True, False]).index
pool["운용순위"] = 0
pool.loc[ord_idx, "운용순위"] = np.arange(1, n + 1)
pool["운용티어"] = "-"
for k, t in [(2000, "C"), (1000, "B"), (500, "A")]:
    pool.loc[pool["운용순위"] <= k, "운용티어"] = t
pool["실제2025가입"] = y

# ── 6. CSV 산출 (전체 1종 + TOP50 1종, 동일 열) ─────────────────
cols = ["운용순위", "운용티어", "customer_id", "score", "예외사유",
        "age", "marital_status", "gender", "region", "income_bracket",
        "risk_grade", "credit_rating", "rider", "cnt2", "tenure", "실제2025가입"]
rename = {"age": "연령대", "marital_status": "결혼", "gender": "성별",
          "region": "지역", "income_bracket": "소득구간", "risk_grade": "위험등급",
          "credit_rating": "신용등급", "rider": "특약보유", "cnt2": "보유수2이상",
          "tenure": "경과연수"}
out = pool[cols].sort_values("운용순위").copy()
out = out.rename(columns=rename)
out["score"] = out["score"].round(4)
for c in ["특약보유", "보유수2이상", "경과연수", "실제2025가입"]:
    out[c] = out[c].astype(int)
csv_path = os.path.join(OUT, "상담대상_순위_T2024.csv")
out.to_csv(csv_path, index=False, encoding="utf-8-sig")
top50_path = os.path.join(OUT, "상담대상_TOP50_T2024.csv")
out.head(50).to_csv(top50_path, index=False, encoding="utf-8-sig")
print(f"[CSV1] {csv_path} ({os.path.getsize(csv_path):,} bytes)")
print(f"[CSV2] {top50_path} ({os.path.getsize(top50_path):,} bytes)")

# ── 7. 콘솔 보고용 수치 ────────────────────────────────────────
print("\n[운용순위 기준 실제 가입률 — 티어 경계 500/1,000/2,000]")
cap = {}
for k in (500, 1000, 2000):
    m = (pool["운용순위"] <= k).values
    cap[k] = (int(y[m].sum()), y[m].mean() * 100)
    print(f"  상위 {k:,}명 : 실제 가입률 {y[m].mean()*100:.1f}%  "
          f"포착 양성 {y[m].sum():,}명 ({y[m].sum()/pos*100:.0f}% of 양성)")

exc_total = int((pool["_grp"] > 0).sum())
exc_only_r = int((ex_region & ~(ex_m20 | ex_c300)).sum())
exc_only_m = int((ex_m20 & ~ex_region & ~ex_c300).sum())
exc_only_c = int((ex_c300 & ~ex_region & ~ex_m20).sum())
exc_r_m = int((ex_region & ex_m20 & ~ex_c300).sum())
exc_r_c = int((ex_region & ex_c300 & ~ex_m20).sum())
exc_m_c = int((ex_m20 & ex_c300 & ~ex_region).sum())
exc_all3 = int((ex_region & ex_m20 & ex_c300).sum())
print(f"\n[예외 총 {exc_total:,}명 (풀의 {exc_total/n*100:.1f}%)] "
      f"저율지역 {int(ex_region.sum()):,} / 20대기혼 {int(ex_m20.sum()):,} / "
      f"신용300 {int(ex_c300.sum()):,} (중복 포함, 유형별 합계)")
print(f"  조합별: 단독(지역/기혼/300)={exc_only_r:,}/{exc_only_m:,}/{exc_only_c:,}  "
      f"지역+기혼={exc_r_m:,}  지역+300={exc_r_c:,}  기혼+300={exc_m_c:,}  셋모두={exc_all3:,}")

top3 = out.head(3)
print("\n[TOP50 첫 3행]")
print(top3.to_string(index=False))

# ── 8. HTML 보드 ───────────────────────────────────────────────
def esc(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)

rows50 = []
for _, r in out.head(50).iterrows():
    rows50.append(
        f"<tr><td>{int(r['운용순위']):,}</td><td>{esc(r['운용티어'])}</td>"
        f"<td class='l'>{esc(r['customer_id'])}</td><td>{r['score']:.4f}</td>"
        f"<td class='l sm'>{esc(r['예외사유'])}</td><td>{esc(r['연령대'])}</td>"
        f"<td>{esc(r['결혼'])}</td><td>{esc(r['성별'])}</td><td>{esc(r['지역'])}</td>"
        f"<td>{esc(r['소득구간'])}</td><td>{esc(r['위험등급'])}</td>"
        f"<td>{esc(r['신용등급'])}</td><td>{'✓' if r['특약보유'] else ''}</td>"
        f"<td>{'✓' if r['보유수2이상'] else ''}</td><td>{int(r['경과연수'])}</td>"
        f"<td class='ok'>{'✓' if r['실제2025가입'] else ''}</td></tr>")
table50 = "\n".join(rows50)

top500 = pool[pool["운용순위"] <= 500]
def share(df, col, val=None):
    s = df[col].eq(val) if val is not None else df[col].astype(bool)
    return s.mean() * 100
inc5_t = top500["income_bracket"].isin(INC_BELOW5000).mean() * 100
inc5_p = pool["income_bracket"].isin(INC_BELOW5000).mean() * 100
comp_rows = [
    ("특약보유", share(top500, "rider"), share(pool, "rider")),
    ("보유수2이상", share(top500, "cnt2"), share(pool, "cnt2")),
    ("기혼", share(top500, "marital_status", "기혼"), share(pool, "marital_status", "기혼")),
    ("여성", share(top500, "gender", "여"), share(pool, "gender", "여")),
    ("소득 5,000만원 미만", inc5_t, inc5_p),
]
comp_html = "\n".join(
    f"<tr><td class='l'>{lab}</td><td>{a:.1f}%</td><td>{b:.1f}%</td>"
    f"<td>{a-b:+.1f}%p</td></tr>" for lab, a, b in comp_rows)

cap_html = "\n".join(
    f"<li>상위 {k:,}명(티어 {'A' if k==500 else 'B' if k==1000 else 'C'}): "
    f"실제 가입 {cap[k][0]:,}명 — 실제 가입률 {cap[k][1]:.1f}%</li>"
    for k in (500, 1000, 2000))

html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>상담 대상 순위 보드 — T=2024년 말 풀</title>
<style>
  body {{ font-family:'Malgun Gothic','맑은 고딕',sans-serif; margin:24px auto; max-width:1180px;
         color:#232a33; background:#EEF1F4; padding:0 16px; }}
  h1 {{ font-size:22px; color:#16307a; margin:0 0 4px; }}
  h2 {{ font-size:15px; color:#16307a; margin:0 0 8px; }}
  .meta {{ color:#5b6572; font-size:12.5px; margin-bottom:16px; }}
  .tiles {{ display:flex; gap:12px; margin-bottom:16px; flex-wrap:wrap; }}
  .tile {{ background:#fff; border:1px solid #d6dce3; border-radius:6px;
          padding:12px 18px; min-width:180px; }}
  .num {{ font-size:23px; font-weight:bold; color:#16307a; }}
  .lab {{ font-size:12px; color:#5b6572; margin-top:4px; }}
  .box {{ background:#fff; border:1px solid #d6dce3; border-radius:6px;
         padding:13px 17px; margin-bottom:16px; font-size:13px; line-height:1.65; }}
  table {{ border-collapse:collapse; width:100%; background:#fff; font-size:12px; margin-bottom:6px; }}
  th, td {{ border:1px solid #ccd3db; padding:5px 7px; text-align:center; }}
  th {{ background:#e3e9f0; color:#16307a; }}
  td.l {{ text-align:left; }}
  td.sm {{ font-size:11px; color:#5b6572; }}
  .ok {{ color:#1a7f37; font-weight:bold; }}
  .note {{ font-size:12px; color:#7a5200; background:#fdf6e3; border:1px solid #e8d9a0;
          border-radius:6px; padding:10px 14px; margin-bottom:16px; line-height:1.6; }}
  ul {{ margin:4px 0 0 18px; padding:0; }}
  .foot {{ font-size:11.5px; color:#5b6572; margin-top:10px; line-height:1.6; }}
</style>
</head>
<body>
<h1>상담 대상 순위 보드 — T=2024년 말 풀</h1>
<div class="meta">모델 v3 · 로지스틱 회귀(max_iter=2000 · class_weight=balanced · random_state=42) ·
피처 13계열(범주 8 원핫 + 수치 5 + 연령×기혼 교차) · 2026-10-16 생성</div>

<div class="box">
<h2>기준</h2>
<b>풀</b> : 2024년(T) 이전에 가입한 보험 상품을 1개 이상 보유한 고객 {n:,}명(고객 단위 1행)<br>
<b>타깃</b> : 2025년(T 이후) 새 상품을 1개 이상 실제 가입하면 1 — 양성 {pos:,}명({pos/n*100:.1f}%)<br>
<b>score</b> : 전체 풀로 학습한 로지스틱 회귀의 2025년 가입 확률 예측값(높을수록 유망) —
전체 풀 학습·전체 풀 스코어(in-sample)이므로 검증·운용 참고용<br>
<b>운용순위</b> : score 내림차순이되 예외 3종(아래 운용 규칙)을 뒤로 민 순위
</div>

<div class="tiles">
  <div class="tile"><div class="num">{n:,}</div><div class="lab">풀(보험 고객, T=2024년 말 보유)</div></div>
  <div class="tile"><div class="num">{pos:,}</div><div class="lab">양성(2025년 실제 신규 가입)</div></div>
  <div class="tile"><div class="num">{pos/n*100:.1f}%</div><div class="lab">양성 비율(기준선)</div></div>
  <div class="tile"><div class="num">{exc_total:,}</div><div class="lab">예외 후순위 대상(3종 합산, 중복 포함)</div></div>
</div>

<h2>TOP 50 (운용순위 기준)</h2>
<table>
<tr><th>운용순위</th><th>운용티어</th><th>customer_id</th><th>score</th><th>예외사유</th>
<th>연령대</th><th>결혼</th><th>성별</th><th>지역</th><th>소득구간</th><th>위험등급</th>
<th>신용등급</th><th>특약보유</th><th>보유수2이상</th><th>경과연수</th>
<th>실제2025가입(검증용·관측됨)</th></tr>
{table50}
</table>
<div class="cap" style="font-size:12px;color:#5b6572;margin-bottom:18px;">
맨 오른쪽 실제2025가입 열은 이미 관측된 결과(검증용·관측됨) — 실제 운용 시점에는 알 수 없는 값.</div>

<h2>상위 500명 구성 — 전체 풀과 비교</h2>
<table>
<tr><th>구성</th><th>TOP 500</th><th>전체 풀</th><th>차이</th></tr>
{comp_html}
</table>
<div class="cap" style="font-size:12px;color:#5b6572;margin-bottom:16px;">
소득 5,000만원 미만 = 1000·3000·5000만원미만 구간 합계.</div>

<div class="box">
<h2>티어별 실제 가입률 (운용순위 기준 — 참고용)</h2>
<ul>
{cap_html}
</ul>
<div class="note"><b>유의</b> : 이 수치는 전체 풀로 학습한 스코어를 같은 풀에서 다시 세어본 것(in-sample)이라
평가세트 수치(상위 500/1,000/2,000 = 9.7/8.2/6.7%)보다 높게 나올 수 있음 — 참고용으로만 볼 것.</div>
</div>

<div class="box">
<h2>운용 규칙</h2>
<b>운용티어</b> : 운용순위 1~500 = <b>A</b> · 501~1,000 = <b>B</b> · 1,001~2,000 = <b>C</b> · 그 외 = <b>-</b><br>
<b>예외 후순위 3종</b> : ① 지역이 대구·대전·인천(저율지역 후순위) ② 20대이면서 기혼(20대 기혼 후순위)
③ 신용 '300점대'(신용300 최후순위)<br>
<b>적용 방식</b> : 예외 없는 고객을 score 내림차순으로 1~N 배치 → ①② 예외 고객을 score순으로 그 뒤에 →
신용300(③)은 맨 마지막에 score순 배치. 예외가 겹치면 사유를 모두 표기하고, 신용300이 있으면 맨 마지막 그룹으로.
</div>

<div class="foot">
<b>재현 경로</b> : python scripts/step4_rank_list.py<br>
데이터 : [금융] 금융상품·서비스 및 소비자 특성 데이터/Training/02.라벨링데이터/TL_2.소비자.zip (utf-8-sig) ·
산출물 : output/상담대상_순위_T2024.csv · output/상담대상_TOP50_T2024.csv · output/상담대상_순위보드_T2024.html
</div>
</body>
</html>
"""
html_path = os.path.join(OUT, "상담대상_순위보드_T2024.html")
with open(html_path, "w", encoding="utf-8-sig") as f:
    f.write(html)
print(f"[HTML] {html_path} ({os.path.getsize(html_path):,} bytes)")
print("[완료] 산출물 3종 저장 (순위 CSV + TOP50 CSV + HTML 보드)")
