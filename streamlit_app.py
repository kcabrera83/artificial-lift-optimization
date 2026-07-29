import streamlit as st, joblib, numpy as np, matplotlib.pyplot as plt
from pathlib import Path; import sys; sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Artificial Lift Optimization", layout="wide")
st.title("Artificial Lift Optimization")

p = Path(__file__).parent / 'outputs' / 'models'
models = {'optimizer': joblib.load(p / 'lift_optimizer.pkl')}

tab1, tab2, tab3 = st.tabs(['Predict', 'Charts', 'Info'])

with tab1:
    st.subheader('Inputs')
    c = st.columns(2)
    lift = c[0].selectbox('Lift', ['ESP','gas_lift','rod_pump'])
    speed = c[1].slider('Speed', 500, 4000, 2250)
    depth = c[0].slider('Depth', 1000, 15000, 8000)
    wcut = c[1].slider('Wcut', 0, 100, 50)
    pressure = c[0].slider('Pressure', 500, 5000, 2750)
    current = c[1].slider('Current', 10, 200, 105)
    if st.button('Run', type='primary'):
        x = np.array([[lift, speed, depth, wcut, pressure, current]])
        res = {}
        m = models['optimizer']
        if isinstance(m, dict):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            res['optimizer'] = m['label_encoder'].inverse_transform(p)[0] if 'label_encoder' in m else float(p[0])
        else:
            res['optimizer'] = float(m.predict(x)[0])
        st.divider()
        rc = st.columns(len(res))
        for i, (k, v) in enumerate(res.items()):
            rc[i].metric(k.replace('_',' ').title(), str(v) if isinstance(v,str) else f'{v:.2f}')

with tab2:
    st.info('Charts update after prediction')

with tab3:
    st.markdown('ESP / Rod Pump / Gas Lift optimization')
    st.caption('Built with scikit-learn + Streamlit')