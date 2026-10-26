# -*- coding: utf-8 -*-
t = open("_v_log8.txt", encoding="utf-8").read()
bad = ("불일치" in t) or ("FAIL" in t) or ("Error" in t) or ("Traceback" in t)
if bad:
    raise SystemExit(1)
print("VERIFY_PASS lines=%d" % t.count("\n"))