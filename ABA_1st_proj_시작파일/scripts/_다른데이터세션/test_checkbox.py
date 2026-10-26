from streamlit.testing.v1 import AppTest

path = r"C:\Users\user\Desktop\실습8 앱으로 만들기\app.py"

at = AppTest.from_file(path, default_timeout=60).run()
print("default:", at.exception, "metric:", [m.value for m in at.metric])

at.button[0].click()
at.run()
print("clicked, checkbox off:", at.exception, [m.value for m in at.metric])

at.sidebar.checkbox[0].check()
at.button[0].click()
at.run()
print("clicked, checkbox on:", at.exception,
      [m.label for m in at.metric], [m.value for m in at.metric])
