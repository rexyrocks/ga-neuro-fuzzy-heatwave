from pathlib import Path
import pandas as pd
import streamlit as st
from heatwave import FEATURES, predict_file
st.set_page_config(page_title='Heatwave Risk Explorer', page_icon='☀️')
st.title('Heatwave Risk Explorer')
st.caption('GA-optimized neuro-fuzzy research demonstration')
root = Path(__file__).parent
mode = st.selectbox('Dataset / model', ['New Delhi historical weather', 'Synthetic demonstration'])
real = mode == 'New Delhi historical weather'
model = root / ('results_delhi' if real else 'results') / 'model.pkl'
if real:
    st.warning('NASA POWER gridded weather, 2015–2025. Categories are research-derived temperature hazard labels, not official alerts or observed health outcomes.')
    with st.expander('Historical dataset and evaluation'):
        history = pd.read_csv(root / 'data/new_delhi/delhi_2015_2025_labeled.csv')
        st.caption('New Delhi: 28.6139°N, 77.2090°E. Reference climatology: 1985–2014. Train: 2015–2021; validation: 2022–2023; test: 2024–2025.')
        st.line_chart(history.set_index('date')[['max_temp_c', 'min_temp_c']])
        st.dataframe(history['risk'].value_counts().rename('Days'))
        st.download_button('Download historical dataset', history.to_csv(index=False), 'delhi_2015_2025.csv', 'text/csv')
        st.caption('Low: below research thresholds. Moderate: at least local p90 or 38°C. High: IMD-like heat-day temperature criteria. Extreme: IMD-like severe-day criteria OR at least 40°C and local p97.5. This four-class mapping is invented for research.')
        metrics_file = root / 'results_delhi/metrics.json'
        if metrics_file.exists():
            import json
            metrics = json.loads(metrics_file.read_text())['models']
            st.dataframe(pd.DataFrame({name: {'Accuracy': m['accuracy'], 'Macro F1': m['macro_f1']} for name, m in metrics.items()}).T)
else:
    st.warning('Synthetic demonstration with invented labels; not official alerts or medical advice.')
if not model.exists():
    st.info('Run python heatwave.py demo first to create the model.'); st.stop()
with st.form('weather'):
    maximum = st.number_input('Maximum temperature (°C)', value=41.)
    minimum = st.number_input('Minimum temperature (°C)', value=29.)
    humidity = st.slider('Relative humidity (%)', 0, 100, 60)
    duration = st.number_input('Consecutive days above local p90 (use 1 if none)', min_value=1, value=4)
    anomaly = st.number_input('Temperature anomaly relative to local climatology (°C)', value=5.)
    submit = st.form_submit_button('Classify risk')
if submit:
    try:
        result = predict_file(model, pd.DataFrame([[maximum, minimum, humidity, duration, anomaly]], columns=FEATURES))
        st.subheader(result.predicted_risk.iloc[0])
        st.bar_chart(result.filter(like='probability_').iloc[0].rename(index=lambda x: x.replace('probability_', '').title()))
        st.caption('Model scores are not calibrated probabilities of illness or death.')
    except ValueError as error: st.error(str(error))
upload = st.file_uploader('Batch prediction CSV', type='csv')
if upload is not None:
    try:
        result = predict_file(model, pd.read_csv(upload)); st.dataframe(result)
        st.download_button('Download predictions', result.to_csv(index=False), 'predictions.csv', 'text/csv')
    except (ValueError, KeyError) as error: st.error(str(error))
