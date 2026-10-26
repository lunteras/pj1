# -*- coding: utf-8 -*-
import shutil, pathlib
src = pathlib.Path(r"C:\Users\user\AppData\Local\Temp\helpycode\worksheet.pdf")
dst = pathlib.Path(r"C:\Users\user\Downloads") / "주제선정세션_워크시트.pdf"
shutil.copyfile(src, dst)
print("copied:", dst.exists(), dst.stat().st_size)
