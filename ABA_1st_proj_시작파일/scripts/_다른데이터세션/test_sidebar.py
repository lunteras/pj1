from streamlit.testing.v1 import AppTest

path = r"C:\Users\user\Desktop\실습8 앱으로 만들기\app.py"

at = AppTest.from_file(path, default_timeout=60).run()
print("widgets:", at.sidebar.slider[0].label, at.sidebar.radio[0].label,
      at.sidebar.number_input[0].label, at.sidebar.radio[1].label,
      at.sidebar.selectbox[0].label)
print("before click:", at.exception, "metric:", [m.value for m in at.metric])

at.button[0].click()
at.run()
print("after click:", at.exception, "metric:", [m.value for m in at.metric])

at.sidebar.slider[0].set_value(50)
at.run()
print("age=50:", at.exception, "metric:", [m.value for m in at.metric])
