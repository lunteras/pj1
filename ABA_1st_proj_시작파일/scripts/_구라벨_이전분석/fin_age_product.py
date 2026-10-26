import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

fin = pd.read_csv(
    r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip",
    encoding="utf-8-sig",
)

sec1 = fin[(fin["customer_type"] == "증권") & (fin["customer_seq"] == 1)].copy()

# 연령대 순서 지정
age_order = ["20대", "30대", "40대", "50대", "60대", "70대", "80대 이상"]

# 상품 그룹화: 채권+해외채권 / 소수 상품은 기타로 합침
def grp(k):
    if k == "해외채권":
        return "채권(해외 포함)"
    if k == "채권":
        return "채권(해외 포함)"
    if k in ["해외RP", "해외발행어음", "ELS/DLS", "개인연금"]:
        return "기타"
    return k

sec1["상품군"] = sec1["transaction_keyword"].map(grp)

col_order = ["RP", "MMW", "발행어음", "퇴직연금", "펀드", "채권(해외 포함)", "기타"]

ct = pd.crosstab(sec1["age"], sec1["상품군"])[col_order]
ct = ct.reindex(age_order)
pct = ct.div(ct.sum(axis=1), axis=0) * 100
pct = pct.round(1)

out = pd.DataFrame({"행 수": ct.sum(axis=1)})
for c in col_order:
    out[c + "(%)"] = pct[c]
out["안전자산3종(%)"] = (pct["RP"] + pct["MMW"] + pct["발행어음"]).round(1)

print(out.to_string())
print()
print("안전자산 3종 = RP + MMW + 발행어음 (예금과 비슷하거나 낮은 위험)")
print()
# 거른 행 확인 (미수집/미답변/- 등)
blank_age = sec1["age"].isin(["미수집", "미답변", "-", "기타"]).sum()
blank_kw = sec1["transaction_keyword"].isin(["미수집", "미답변", "-", "기타"]).sum()
print("거른 행: age 빈칸", blank_age, "건 / 상품 빈칸", blank_kw, "건")