# -*- coding: utf-8 -*-
import re, sys, io, pathlib, unicodedata
import markdown

# Downloads의 파일명이 NFD(분해형 자모)일 수 있어 실제 파일을 찾아 쓴다
DL = pathlib.Path(r"C:\Users\user\Downloads")
SRC = None
for p in DL.glob("*.md"):
    if unicodedata.normalize("NFC", p.name) == "주제선정세션_워크시트.md":
        SRC = p
        break
assert SRC is not None, "worksheet md not found"
OUT_HTML = pathlib.Path(r"C:\Users\user\AppData\Local\Temp\helpycode\worksheet.html")
OUT_PDF = DL / "주제선정세션_워크시트.pdf"

text = SRC.read_text(encoding="utf-8")

# YAML frontmatter 분리
meta = {}
m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
if m:
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    text = text[m.end():]

# 없는 SVG 이미지 → 안내 박스로 대체
def img_repl(mo):
    alt = mo.group(1)
    return ('<div class="img-placeholder">[ 그림 생략: %s ]</div>' % alt)

text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", img_repl, text)

html_body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])

css = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: "Malgun Gothic", "Segoe UI Symbol", sans-serif;
  font-size: 10.5pt; line-height: 1.65; color: #1a1a1a; margin: 0;
}
h1 {
  font-size: 19pt; border-bottom: 2.5px solid #2b3a55; padding-bottom: 8px;
  margin: 0 0 6px 0; color: #2b3a55;
}
.doc-meta { color: #666; font-size: 9pt; margin-bottom: 22px; }
h2 {
  font-size: 14pt; color: #2b3a55; border-left: 5px solid #2b3a55;
  padding-left: 8px; margin: 26px 0 10px 0;
  page-break-after: avoid; break-after: avoid;
}
h3 {
  font-size: 12pt; color: #34455f; margin: 20px 0 8px 0;
  page-break-after: avoid; break-after: avoid;
}
p { margin: 6px 0; }
ul, ol { margin: 6px 0; padding-left: 24px; }
li { margin: 3px 0; }
table {
  border-collapse: collapse; width: 100%; margin: 10px 0 14px 0;
  font-size: 9.5pt; page-break-inside: avoid;
}
th {
  background: #eef1f6; border: 1px solid #b9c2d0; padding: 5px 7px;
  text-align: left; font-weight: 600;
}
td { border: 1px solid #c9d0da; padding: 5px 7px; vertical-align: top; }
tr:nth-child(even) td { background: #f8f9fb; }
pre {
  background: #f4f5f7; border: 1px solid #d8dce2; border-radius: 4px;
  padding: 10px 12px; font-family: Consolas, "Malgun Gothic", monospace;
  font-size: 9.5pt; white-space: pre-wrap; word-wrap: break-word;
  page-break-inside: avoid; margin: 8px 0;
}
code {
  font-family: Consolas, "Malgun Gothic", monospace; font-size: 9.5pt;
  background: #f0f1f3; padding: 1px 4px; border-radius: 3px;
}
pre code { background: none; padding: 0; }
blockquote {
  border-left: 4px solid #b9c2d0; margin: 10px 0; padding: 2px 12px;
  color: #444; background: #f8f9fb;
}
hr { border: none; border-top: 1.5px solid #d0d5dd; margin: 20px 0; }
strong { color: #000; }
.img-placeholder {
  border: 1.5px dashed #aab3c0; border-radius: 4px; background: #fafbfc;
  color: #6b7686; font-size: 9pt; text-align: center;
  padding: 9px 10px; margin: 10px 0;
}
a { color: #2b5aa0; text-decoration: none; word-break: break-all; }
"""

title = meta.get("title", "1차 프로젝트 주제 고르기 워크시트")
date = meta.get("date", "")
doc_meta = " · ".join(x for x in [meta.get("type", ""), date] if x)

html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style></head>
<body>
<h1>{title}</h1>
<div class="doc-meta">{doc_meta}</div>
{html_body}
</body></html>"""

OUT_HTML.write_text(html, encoding="utf-8")
print("HTML written:", OUT_HTML)
print("PDF target:", OUT_PDF)
