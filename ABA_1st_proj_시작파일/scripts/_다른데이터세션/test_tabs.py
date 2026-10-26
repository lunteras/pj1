from streamlit.testing.v1 import AppTest

path = r"C:\Users\user\Desktop\실습8 앱으로 만들기\app.py"

at = AppTest.from_file(path, default_timeout=60).run()
print("tabs:", [t.label for t in at.tabs])

at.button[0].click()
at.run()
print("predict tab:", at.exception,
      "metric:", [m.value for m in at.metric])

print("data tab:", at.exception,
      "dataframe rows:", len(at.dataframe[0].value))
print(at.dataframe[0].value.head(3).to_string(index=False))
