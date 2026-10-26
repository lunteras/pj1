from streamlit.testing.v1 import AppTest

path = r"C:\Users\user\Desktop\실습8 앱으로 만들기\app.py"

at = AppTest.from_file(path, default_timeout=60).run()
print("default(31.0):", at.exception, [m.value for m in at.metric])

at.number_input[0].set_value(32.0)
at.run()
print("BMI=32:", at.exception, [m.value for m in at.metric])
