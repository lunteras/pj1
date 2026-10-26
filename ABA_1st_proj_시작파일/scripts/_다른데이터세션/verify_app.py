from streamlit.testing.v1 import AppTest

def run_with(**label_values):
    at = AppTest.from_file('app.py', default_timeout=60).run()
    for label, v in label_values.items():
        [n for n in at.number_input if n.label == label][0].set_value(v)
    at.run()
    return at

def warn(at):
    return [w.value for w in at.warning]

at = AppTest.from_file('app.py', default_timeout=60).run()
checks = {
    'no_exception': not bool(at.exception),
    'default_no_warning': len(at.warning) == 0,
    'range_caption': any('학습한 범위' in c.value for c in at.caption),
    'metrics_ok': len(at.metric) == 2 and at.metric[0].proto.body == '39,540달러',
    'chart_present': len(at.get('arrow_vega_lite_chart')) == 1,
}

w = warn(run_with(**{'BMI': 60}))
checks['bmi60_warn'] = len(w) == 1 and 'BMI 60' in w[0] and '믿기 어렵' in w[0]
w = warn(run_with(**{'BMI': 53.13}))
checks['bmi_max_ok'] = len(w) == 0
w = warn(run_with(**{'BMI': 53.5}))
checks['bmi_over_warn'] = len(w) == 1
w = warn(run_with(**{'나이': 70}))
checks['age70_warn'] = len(w) == 1 and '나이 70' in w[0]
w = warn(run_with(**{'나이': 64}))
checks['age_max_ok'] = len(w) == 0
w = warn(run_with(**{'나이': 65}))
checks['age_over_warn'] = len(w) == 1
w = warn(run_with(**{'자녀 수': 7}))
checks['children7_warn'] = len(w) == 1 and '자녀 수 7' in w[0]
w = warn(run_with(**{'자녀 수': 5}))
checks['children_max_ok'] = len(w) == 0
at3 = AppTest.from_file('app.py', default_timeout=60).run()
[n for n in at3.number_input if n.label == '나이'][0].set_value(70)
[n for n in at3.number_input if n.label == 'BMI'][0].set_value(60)
at3.run()
w3 = warn(at3)
checks['multi_warn'] = len(w3) == 1 and '나이 70' in w3[0] and 'BMI 60' in w3[0]

for k in sorted(checks):
    print(k, checks[k])
print('ALL_OK', all(checks.values()))
