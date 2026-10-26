import importlib.metadata as m

for p in ['streamlit', 'scikit-learn', 'numpy', 'pandas', 'joblib']:
    try:
        print(f'{p}=={m.version(p)}')
    except m.PackageNotFoundError:
        print(f'{p}==NOT INSTALLED')
