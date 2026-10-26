import re
p = r"C:\Users\user\Desktop\ABA_1st_proj_시작파일_v2\artifacts\보장only_전체정리_입문.html"
h = open(p, encoding="utf-8").read()
print("sections:", h.count("<section"), "closes:", h.count("</section>"),
      "svg:", h.count("<svg"), "ext_url:", ("http" in h),
      "chips:", h.count("c-ok") + h.count("c-bad"),
      "tables:", h.count("class=\"tbl\""))