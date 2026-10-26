# -*- coding: utf-8 -*-
"""EDA 리포트 상세판 조립기 — 조각 3개 + 그림(base64 임베드)으로 완전 자체 포함 HTML 생성
산출: artifacts/보장only_EDA_리포트_상세.html
"""
import os, base64, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ART = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(ART, "..", "ABA_1st_proj_시작파일", "output")
OUTD = os.path.normpath(OUTD)
parts = []
for name in ["_frag1.html", "_frag2.html", "_frag3.html"]:
    parts.append(io.open(os.path.join(ART, name), encoding="utf-8").read())
html = "".join(parts)
KEYS = ["기타편향", "수치형분포", "타겟분포", "상관히트맵", "소득", "지역", "특약", "가설EF"]
pngs = {}
for fn in sorted(os.listdir(OUTD)):
    if fn.lower().endswith(".png"):
        pngs[fn] = fn
def find(key):
    best = None
    for fn in sorted(pngs):
        if key in fn:
            best = fn; break
    return best
n = 0
for key in KEYS:
    token = "{{FIG:" + key + "}}"
    if token not in html:
        continue
    fn = find(key)
    if fn is None:
        rep = "<figure><figcaption><b>" + key + "</b> — 그림 파일을 output에서 찾지 못함</figcaption></figure>"
    else:
        b64 = base64.b64encode(io.open(os.path.join(OUTD, fn), "rb").read()).decode()
        rep = ("<figure><img src='data:image/png;base64," + b64 + "'>" + "<figcaption><b>" + fn + "</b> — output 폴더 원본 그림</figcaption></figure>")
        n += 1
    html = html.replace(token, rep, 1)
left = html.count("{{FIG:")
out = os.path.join(ART, "보장only_EDA_리포트_상세.html")
io.open(out, "w", encoding="utf-8").write(html)
print("ASSEMBLED figs=%d tokens_left=%d KB=%d" % (n, left, len(html.encode("utf-8")) // 1024))
if left > 0:
    raise SystemExit(1)