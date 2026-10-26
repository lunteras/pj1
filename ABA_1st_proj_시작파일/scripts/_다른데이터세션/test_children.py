from streamlit.testing.v1 import AppTest

path = r"C:\Users\user\Desktop\실습8 앱으로 만들기\app.py"

at = AppTest.from_file(path, default_timeout=60).run()
print("sliders:", [w.label for w in at.sidebar.slider],
      "radios:", [w.label for w in at.sidebar.radio])
print("default:", at.exception, "metric:", [m.value for m in at.metric])

at.button[0].click()
at.run()
print("children=1 clicked:", at.exception, [m.value for m in at.metric])

at.sidebar.slider[1].set_value(3)
at.run()
print("children=3 not clicked:", at.exception, [m.value for m in at.metric])
at.button[0].click()
at.run()
print("children=3 clicked:", at.exception, [m.value for m in at.metric])
