from streamlit.testing.v1 import AppTest

path = r"C:\Users\user\Desktop\실습8 앱으로 만들기\app.py"

at = AppTest.from_file(path, default_timeout=60).run()
print("default(35):", at.exception, [m.value for m in at.metric])

at.slider[0].set_value(40)
at.run()
print("age=40:", at.exception, [m.value for m in at.metric])
