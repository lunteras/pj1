# -*- coding: utf-8 -*-
"""정리 작업: (1) 구라벨 그림 -> output/archive_구라벨 이동 (2) 임시폴더 스크립트 -> scripts 복사 + 목록 생성"""
import os, re, shutil, glob as g

BASE = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\ABA_1st_proj_시작파일"
OUT = os.path.join(BASE, "output")
ARC = os.path.join(OUT, "archive_구라벨")
SCR = os.path.join(BASE, "scripts")
TMP = r"C:\Users\user\AppData\Local\Temp\helpycode"
RECS = os.path.join(BASE, "output")

os.makedirs(ARC, exist_ok=True)
os.makedirs(SCR, exist_ok=True)

# (1) 그림 분류: 유효 = 현재 예측 프레임(T2023/T2024) 그림만, 나머지 = 구라벨·구분석
keep_re = re.compile(r"^(EDA_T2024_|예측_T202[34]_)")
pngs = [f for f in os.listdir(OUT) if f.lower().endswith(".png")]
kept, moved = [], []
for f in sorted(pngs):
    if keep_re.match(f):
        kept.append(f)
    else:
        shutil.move(os.path.join(OUT, f), os.path.join(ARC, f))
        moved.append(f)

with open(os.path.join(ARC, "_이동목록.txt"), "w", encoding="utf-8") as fh:
    fh.write("구라벨(대용라벨 '다중 담보') 기반 그림 — 2026-10-15 이동. "
             "예측 프레임(T=2024, 실제 타겟)에는 사용 금지.\n\n")
    fh.write("\n".join(moved))
with open(os.path.join(RECS, "_유효그림목록.txt"), "w", encoding="utf-8") as fh:
    fh.write("유효 그림(예측 프레임 T=2024 기준, T2023은 교차 대조용):\n\n")
    fh.write("\n".join(kept))

# (2) 스크립트 복사 + 설명 추출
py_files = sorted(g.glob(os.path.join(TMP, "*.py")))
manifest = ["# 스크립트 목록 (보장only 추가가입 예측)\n",
            "원본 위치: C:\\Users\\user\\AppData\\Local\\Temp\\helpycode (복사본)\n",
            "실행 기준 데이터: [금융].../Training/02.라벨링데이터/TL_2.소비자.zip (utf-8-sig)\n"]
for p in py_files:
    name = os.path.basename(p)
    src = open(p, encoding="utf-8", errors="replace").read()
    # 첫 docstring 또는 주석 블록에서 설명 추출
    desc = ""
    m = re.match(r"^# -\*- coding[\s\S]*?\"\"\"(.*?)\"\"\"", src, re.S)
    if m:
        desc = " ".join(m.group(1).split())[:120]
    else:
        lines = [ln[1:].strip() for ln in src.splitlines()[:6] if ln.strip().startswith("#")]
        desc = " ".join(lines)[:120]
    shutil.copy2(p, os.path.join(SCR, name))
    manifest.append(f"- `{name}` — {desc}")

with open(os.path.join(SCR, "스크립트_목록.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(manifest))

print(f"figures kept={len(kept)} moved={len(moved)}")
print(f"scripts copied={len(py_files)}")
print("output root png now:", len([f for f in os.listdir(OUT) if f.lower().endswith('.png')]))
