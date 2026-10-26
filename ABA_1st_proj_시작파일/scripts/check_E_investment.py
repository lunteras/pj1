# -*- coding: utf-8 -*-
"""가설 E 함정 확인 — 투자형(변액·연금) 상품 미보유자 안에서의 특약 신호"""
import io, sys, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
warnings.filterwarnings("ignore")
import pandas as pd

SRC = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일\[금융] 금융상품·서비스 및 소비자 특성 데이터\Training\02.라벨링데이터\TL_2.소비자.zip"
fin = pd.read_csv(SRC, encoding="utf-8-sig")
ins = fin[fin["customer_type"] == "보험"].copy()
cust = ins.drop_duplicates("customer_id").set_index("customer_id")

for T in [2024, 2023]:
    past = ins[ins["transaction_start_year"] <= T]
    past_ids = set(past["customer_id"].unique())
    fut_ids = set(ins.loc[ins["transaction_start_year"] > T, "customer_id"].unique())
    pool = cust[cust.index.isin(past_ids)].copy()
    pool["label"] = pool.index.isin(fut_ids).astype(int)
    prod = past.groupby("customer_id").agg(
        names=("product_name", lambda s: " | ".join(sorted(set(s)))),
        kws=("transaction_keyword", lambda s: " | ".join(sorted(set(map(str, s))))),
    )
    pool = pool.join(prod)

    def e_group(row):
        parts = str(row["kws"]).split(" | ")
        has_rider = any(("주보험" in k) and ("특약" in k) for k in parts)
        has_main = any(("주보험" in k) and ("특약" not in k) for k in parts)
        if has_rider: return "주보험+특약"
        if has_main: return "주보험만"
        return "특약만"
    pool["E그룹"] = pool.apply(e_group, axis=1)
    pool["투자형보유"] = pool["names"].apply(lambda s: any(("변액" in n or "연금" in n) for n in str(s).split(" | ")))

    sub = pool[~pool["투자형보유"]]
    g = sub.groupby("E그룹")["label"].agg(["count", "sum", "mean"])
    g.columns = ["고객 수", "가입자", "가입률(%)"]
    g["가입률(%)"] = (g["가입자"] / g["고객 수"] * 100).round(1)
    order = [x for x in ["주보험+특약", "주보험만", "특약만"] if x in g.index]
    print(f"[T={T}] 투자형(변액·연금) 미보유 고객만 (n={len(sub):,}, 가입률 {sub['label'].mean()*100:.1f}%)")
    print(g.reindex(order).to_string())
    small = [f"{i}={int(r['고객 수'])}" for i, r in g.reindex(order).iterrows() if r["고객 수"] < 30]
    print("30미만 셀: " + (", ".join(small) if small else "없음") + "\n")
