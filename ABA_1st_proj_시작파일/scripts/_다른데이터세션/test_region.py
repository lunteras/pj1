from streamlit.testing.v1 import AppTest

path = r"C:\Users\user\Desktop\실습8 앱으로 만들기\app.py"

at = AppTest.from_file(path, default_timeout=60).run()
print("default(남동부):", at.exception, [m.value for m in at.metric])

at.selectbox[0].set_value("북동부")
at.run()
print("북동부:", at.exception, [m.value for m in at.metric])
