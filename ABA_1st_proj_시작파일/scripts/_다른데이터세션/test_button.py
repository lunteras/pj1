from streamlit.testing.v1 import AppTest

path = r"C:\Users\user\Desktop\실습8 앱으로 만들기\app.py"

at = AppTest.from_file(path, default_timeout=60).run()
print("before click:", at.exception, "metric:", [m.value for m in at.metric], "info:", [i.value for i in at.info])

at.button[0].click()
at.run()
print("after click:", at.exception, "metric:", [m.value for m in at.metric], "info:", [i.value for i in at.info])
