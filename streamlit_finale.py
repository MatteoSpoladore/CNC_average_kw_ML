import pandas as pd
import joblib
import plotly.graph_objs as go
import streamlit as st

st.set_page_config(
    page_title="Previsione CNC",
    layout="wide",
)


@st.cache_data
def load_and_predict_new_model():
    # Carico modello e scaler
    model = joblib.load("modello_finale.joblib")
    scaler_X = joblib.load("scaler_X.joblib")
    scaler_y = joblib.load("scaler_y.joblib")

    # Carico e prepara i dati
    df = pd.read_parquet("./factory_20240131.parquet")
    df = df.drop(columns=["humidity_CNC_10", "temp_CNC_10"])
    df["time_batch_dt"] = pd.to_datetime(df["time_batch"], dayfirst=True)
    df = df.sort_values("time_batch_dt").set_index("time_batch_dt")
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Media CNC
    temp_cols = [col for col in df.columns if "temp_CNC" in col]
    humidity_cols = [col for col in df.columns if "humidity_CNC" in col]
    df["mean_temp_CNC"] = df[temp_cols].mean(axis=1)
    df["mean_humidity_CNC"] = df[humidity_cols].mean(axis=1)

    # Filtro come nel training
    df = df[df["cnc_average_kw"] < 400].dropna()

    # X e y
    X_new = df.drop(columns=["time_batch", "cnc_average_kw"], errors="ignore")
    y_true = df["cnc_average_kw"]

    # Trasformo X e y
    X_scaled = scaler_X.transform(X_new)
    y_true_transformed = scaler_y.transform(y_true.values.reshape(-1, 1)).ravel()

    # Predizione e inverse_transform
    y_pred_transformed = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_transformed.reshape(-1, 1)).ravel()
    y_true_original = scaler_y.inverse_transform(
        y_true_transformed.reshape(-1, 1)
    ).ravel()

    return y_true_original, y_pred, X_new


y_true, y_pred, X_sample = load_and_predict_new_model()

st.subheader("📈 Confronto tra valori reali e predetti (modello normalizzato)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=X_sample.index, y=y_true, mode="lines", name="Valori Reali"))
fig.add_trace(
    go.Scatter(
        x=X_sample.index,
        y=y_pred,
        mode="lines",
        name="Predetti",
        line=dict(dash="dash"),
    )
)

fig.update_layout(
    title="📊 Reale vs Predetto in kW (valori filtrati sotto i 400 kw)",
    xaxis_title="Tempo",
    yaxis_title="CNC Average kW",
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)

st.title("🔌 Previsione Consumo CNC")

st.write(
    """
    - Modello: Linear Regression con PowerTransformer;
    - Target trasformato con Yeo-Johnson (Box-Cox), poi riconvertito;
    - Split: 70% train, 30% test;
    - MAE: 0.5268472067570187
    - RMSE: 0.7067050369093744
    - R² Score: 0.25894579879560686
    
    """
)

# Carica modello e scaler
model = joblib.load("modello_finale.joblib")
scaler_X = joblib.load("scaler_X.joblib")
scaler_y = joblib.load("scaler_y.joblib")

# Lista delle feature da usare
feature_names = [
    "work_stations_average_kw",
    "temp_CNC_1",
    "humidity_CNC_1",
    "temp_CNC_2",
    "humidity_CNC_2",
    "temp_CNC_3",
    "humidity_CNC_3",
    "temp_CNC_4",
    "humidity_CNC_4",
    "temp_CNC_5",
    "humidity_CNC_5",
    "temp_CNC_6",
    "humidity_CNC_6",
    "temp_CNC_7",
    "humidity_CNC_7",
    "temp_CNC_8",
    "humidity_CNC_8",
    "temp_CNC_9",
    "humidity_CNC_9",
    "temp_outside",
    "press_mm_hg_outside",
    "humidity_outside",
    "windspeed_outside",
    "visibility_outside",
    "dewpoint_outside",
    "hour",
    "dayofweek",
    "is_weekend",
]


feature_settings = {
    "work_stations_average_kw": {"min": 0.0, "max": 70.0, "value": 3.8},
    "temp_CNC_1": {"min": 16.79, "max": 26.26, "value": 21.68},
    "humidity_CNC_1": {"min": 27.02, "max": 63.36, "value": 40.26},
    "temp_CNC_2": {"min": 16.10, "max": 29.86, "value": 20.34},
    "humidity_CNC_2": {"min": 20.46, "max": 56.03, "value": 40.42},
    "temp_CNC_3": {"min": 17.20, "max": 29.24, "value": 22.27},
    "humidity_CNC_3": {"min": 22.05, "max": 56.76, "value": 40.27},
    "temp_CNC_4": {"min": 15.10, "max": 26.20, "value": 20.86},
    "humidity_CNC_4": {"min": 27.60, "max": 56.82, "value": 41.46},
    "temp_CNC_5": {"min": 15.33, "max": 25.80, "value": 19.59},
    "humidity_CNC_5": {"min": 29.82, "max": 96.32, "value": 50.95},
    "temp_CNC_6": {"min": -6.07, "max": 28.29, "value": 7.91},
    "humidity_CNC_6": {"min": 1.00, "max": 99.90, "value": 54.61},
    "temp_CNC_7": {"min": 15.39, "max": 26.00, "value": 20.27},
    "humidity_CNC_7": {"min": 23.20, "max": 51.40, "value": 35.39},
    "temp_CNC_8": {"min": 16.31, "max": 27.23, "value": 22.03},
    "humidity_CNC_8": {"min": 29.60, "max": 58.78, "value": 42.94},
    "temp_CNC_9": {"min": 15.97, "max": 28.99, "value": 21.90},
    "humidity_CNC_9": {"min": 29.17, "max": 53.33, "value": 41.55},
    "temp_outside": {"min": -5.0, "max": 26.1, "value": 7.41},
    "press_mm_hg_outside": {"min": 733.0, "max": 765.0, "value": 749.0},
    "humidity_outside": {"min": 24.0, "max": 100.0, "value": 79.75},
    "windspeed_outside": {"min": 0.0, "max": 14.0, "value": 4.04},
    "visibility_outside": {"min": 1.0, "max": 66.0, "value": 38.33},
    "dewpoint_outside": {"min": -6.6, "max": 15.5, "value": 3.76},
    "hour": {"min": 0, "max": 23, "value": 11},
    "dayofweek": {"min": 0, "max": 6, "value": 3},
    "is_weekend": {"min": 0, "max": 1, "value": 0},
}

# UI
st.subheader("Inserisci i valori per ciascuna variabile:")
input_data = []
col1, col2 = st.columns(2)
dayofweek_value = None

for i, feature in enumerate(feature_names):
    setting = feature_settings[feature]

    # Determino se il valore è intero (es. hour, dayofweek, is_weekend)
    is_integer = (
        isinstance(setting["min"], int)
        and isinstance(setting["max"], int)
        and isinstance(setting["value"], int)
    )

    if feature == "is_weekend":
        # Calcolo automatico in base a dayofweek_value
        is_weekend_value = int(dayofweek_value in [5, 6])
        with col2:
            st.slider(
                label=feature + " (auto)",
                min_value=0,
                max_value=1,
                value=is_weekend_value,
                step=1,
                disabled=True,
            )
        input_data.append(is_weekend_value)

    elif feature == "dayofweek":
        with col1 if i % 2 == 0 else col2:
            dayofweek_value = st.slider(
                label=feature,
                min_value=setting["min"],
                max_value=setting["max"],
                value=setting["value"],
                step=1,
            )
        input_data.append(dayofweek_value)

    else:
        with col1 if i % 2 == 0 else col2:
            value = st.slider(
                label=feature,
                min_value=int(setting["min"]) if is_integer else float(setting["min"]),
                max_value=int(setting["max"]) if is_integer else float(setting["max"]),
                value=int(setting["value"]) if is_integer else float(setting["value"]),
                step=1 if is_integer else 0.1,
            )
        input_data.append(value)

# Bottone di previsione
if st.button("🔍 Prevedi consumo"):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    # Calcolo colonne derivate
    temp_cols = [col for col in input_df.columns if "temp_CNC" in col]
    humidity_cols = [col for col in input_df.columns if "humidity_CNC" in col]

    # Aggiungo le colonne derivate necessarie al modello
    input_df["mean_temp_CNC"] = input_df[temp_cols].mean(axis=1)
    input_df["mean_humidity_CNC"] = input_df[humidity_cols].mean(axis=1)
    input_scaled = scaler_X.transform(input_df)
    pred_scaled = model.predict(input_scaled)
    pred_kw = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
    st.success(f"⚡ Consumo previsto: **{pred_kw:.2f} kW**")
