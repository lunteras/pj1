from streamlit.testing.v1 import AppTest

path = r"C:\Users\user\Desktop\실습8 앱으로 만들기\app.py"

at = AppTest.from_file(path, default_timeout=60).run()
at.button[0].click()
at.run()
print("no compare selected:", at.exception,
      "metric:", [m.value for m in at.metric],
      "dataframe:", len(at.dataframe))

at.sidebar.multiselect[0].set_value(["북동부", "북서부", "남동부"])
at.button[0].click()
at.run()
print("3 regions selected:", at.exception,
      "metric:", [m.value for m in at.metric])
print(at.dataframe[0].value.to_string(index=False))
