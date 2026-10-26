# -*- coding: utf-8 -*-
import unicodedata, pathlib
for p in pathlib.Path(r"C:\Users\user\Downloads").glob("*.pdf"):
    n = unicodedata.normalize("NFC", p.name)
    if n == "주제선정세션_워크시트.pdf":
        print("FOUND:", p.name == n, p.stat().st_size)
