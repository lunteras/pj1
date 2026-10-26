# -*- coding: utf-8 -*-
"""모델링 검증 그림 갤러리 빌더 — 모델 관련 그림을 근거 설명 캡션과 함께 자체 포함 HTML로 묶음
산출: output/모델링검증_그림모음.html
"""
import os, base64, io

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")
FIGS = [
    ("모델_리프트곡선_T2024_v3.png", "반복 20회 상위 k 농도 곡선 + 95% 신뢰구간 밴드 + 판정선 5.8%",
     "검증 내용: 상위 1,000명 농도 8.2±0.5% — 이번 재검증 실측 8.19±0.47%로 일치.<br>이 그림이 바꾼 결정: 신뢰구간 하한 7.1% > 판정선 5.8% → <b>모델 도입 판정 충족</b>의 근거."),
    ("모델_계수_T2024_v3.png", "로지스틱 계수 상위 15개 (방향 참고용)",
     "검증 내용: 특약 보유 계수 +0.84 최상 · 보유수2+ 다음 — 소거 실험(기여 1위·2위)과 방향 일치.<br>이 그림이 바꾼 결정: 공선성 3쌍(연령×결혼 0.642 등) 때문에 계수를 인과가 아닌 <b>'방향 참고'로만</b> 읽는 원칙."),
    ("모델_확률분포_T2024_v3.png", "모델이 매긴 확률의 분포",
     "검증 내용: 상위 고객만 확률이 뚜렷이 분리되는지 — 순위화(상위 k 선정)가 가능한 형태인지 확인.<br>이 그림이 바꾼 결정: 확률 상위만 농도가 높은 단조 구조 → <b>티어 운용(상위 k 선정)</b>이 유효한 설계임을 뒷받침."),
    ("RF확정_비교.png", "AUC 3막대 + 95% CI (로지스틱 0.720 · RF기본 0.743 · RF튜닝 0.725)",
     "검증 내용: 12조합×5시드 탐색 → 최고 조합(leaf노드 32·sqrt)×20시드 — 튜닝도 기본 RF(0.743)를 못 넘음.<br>이 그림이 바꾼 결정: 로지스틱 대비 운용 우위 +0.10%p±0.60 유의X → <b>RF 전환 근거 없음, 로지스틱 v3 확정 유지</b>."),
    ("가설EF_검증강도.png", "소거 실험 — 피처를 하나씩 빼고 20회 반복으로 성능 비교",
     "검증 내용: −특약 시 리프트 8.15→7.24%(AUC 0.719→0.682), −교차항 시 기여 0, −보유수2+ 시 7.60%.<br>이 그림이 바꾼 결정: 특약 = <b>모델 기여 1위로 강화</b>, 보유수×납입액 교차항 = <b>기여 0으로 피처 제거</b> (14→13개)."),
    ("다음절차_B3_보정곡선.png", "보정 곡선 — 모델 확률 5분위 vs 실제 가입률",
     "검증 내용: Q1 2.30% → Q5 3.66% 단조 상승 — 모델 확률이 실제 빈도를 반영.<br>이 그림이 바꾼 결정: 운용안의 티어 수치를 상담원에게 <b>'10명 중 1명' 식의 실제 빈도로</b> 안내할 수 있게 됨."),
    ("티어경계_민감도.png", "상위 k별 농도 곡선 (k=250~3,000 · 20회 평균±sd)",
     "검증 내용: 1,000·2,000 경계 절벽 없음(전체 기울기 대비 1.5배·0.5배), 500경계만 국소 경사(2.0배).<br>이 그림이 바꾼 결정: 티어 A(500명)는 <b>경계 ±100명에서 약 1%p 변동 가능</b>을 운용안에 명시 — B·C는 안정으로 유지."),
]
banner = open("_model_verify_log.txt", encoding="utf-8").read().replace("&", "&amp;").replace("<", "&lt;")
h = ["<!DOCTYPE html><html lang='ko'><head><meta charset='utf-8'><title>모델링 검증 그림 모음</title><style>",
     "body{font-family:'Malgun Gothic',sans-serif;max-width:860px;margin:24px auto;padding:0 16px;background:#F7F8FA;color:#1F2D3D}",
     "h1{font-size:22px}h2{font-size:16px;margin:34px 0 6px}",
     ".banner{background:#EEF3F9;border-left:4px solid #2456a6;padding:10px 14px;font-size:13px;white-space:pre-wrap}",
     "figure{margin:20px 0;background:#fff;border:1px solid #D5DEE6;border-radius:10px;padding:14px}",
     "img{max-width:100%;display:block;margin:0 auto}",
     "figcaption{font-size:13px;color:#54637A;margin-top:8px;line-height:1.6}",
     ".why{background:#F2F5F8;border-left:3px solid #2F6B4F;padding:6px 10px;margin-top:6px;font-size:13px}",
     "</style></head><body>",
     "<h1>모델링 검증 그림 모음 — 그림별 근거 설명</h1>",
     "<div class='banner'><b>검증 결과 (scripts\\step3_model_verify.py · 2026-10-18 재실행)</b><br>", banner.replace("\n", "<br>"), "</div>",
     "<p style='font-size:13px;color:#54637A'>EDA 검증과 동일 방식: 확정 스크립트를 재실행해 리포트 수치와 대조한 뒤, 검증에 쓰인 그림 전부를 근거 설명과 함께 모았습니다.</p>"]
for fn, title, why in FIGS:
    p = os.path.join(OUT, fn)
    if not os.path.isfile(p):
        h.append(f"<figure><figcaption><b>{fn}</b> — 파일 없음(아카이브 확인 필요)</figcaption></figure>")
        continue
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    h.append(f"<figure><h2>{title}</h2><img src='data:image/png;base64,{b64}'>")
    h.append(f"<figcaption><b>{fn}</b><br>{why}</figcaption></figure>")
h.append("</body></html>")
io.open(os.path.join(OUT, "모델링검증_그림모음.html"), "w", encoding="utf-8").write("".join(h))
print("GALLERY_WRITTEN")