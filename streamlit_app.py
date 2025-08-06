import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
import streamlit as st import pandas as pd import numpy as np import random import matplotlib.pyplot as plt from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor from sklearn.svm import SVR from sklearn.tree import DecisionTreeRegressor from sklearn.neighbors import KNeighborsRegressor from statsmodels.tsa.arima.model import ARIMA import xgboost as xgb import lightgbm as lgb import catboost as cb from keras.models import Sequential from keras.layers import LSTM, Dense import tensorflow as tf

st.set_page_config(page_title="Crash Predictor", layout="centered") st.title("🚀 Crash Multiplier Predictor")

Upload CSV

uploaded_file = st.file_uploader("👉 Upload your crash data (CSV with 'Multiplier(Crash)' column)", type="csv")

@st.cache_data def load_data(file): data = pd.read_csv(file) data['Multiplier(Crash)'] = pd.to_numeric(data['Multiplier(Crash)'], errors='coerce') return data.dropna()

Prediction functions

def prepare_xy(data): x = np.array(range(len(data))).reshape(-1, 1) y = data.values return x, y

def monte_carlo(data): mean = data.mean() std = data.std() sim = [random.gauss(mean, std) for _ in range(10000)] return sum(sim) / len(sim), min(max((1 / std) * 100, 0), 100)

def linear_regression(data): x, y = prepare_xy(data) model = LinearRegression().fit(x, y) pred = model.predict(np.array([[len(data)]]))[0] return pred, model.score(x, y) * 100

def random_forest(data): x, y = prepare_xy(data) model = RandomForestRegressor().fit(x, y) pred = model.predict(np.array([[len(data)]]))[0] return pred, model.score(x, y) * 100

def lstm_model(data): x = np.array(range(len(data))).reshape(-1, 1) y = data.values x_lstm = x.reshape((x.shape[0], 1, 1)) model = Sequential() model.add(LSTM(50, input_shape=(1, 1))) model.add(Dense(1)) model.compile(loss='mean_squared_error', optimizer='adam') model.fit(x_lstm, y, epochs=10, batch_size=1, verbose=0) pred = model.predict(np.array([[[len(data)]]]))[0][0] return pred, 100

def arima_model(data): model = ARIMA(data, order=(5, 1, 0)) model_fit = model.fit() pred = model_fit.forecast()[0] return pred, 100

models = { "Monte Carlo": monte_carlo, "Linear Regression": linear_regression, "Random Forest": random_forest, "LSTM Neural Net": lstm_model, "ARIMA": arima_model }

if uploaded_file: df = load_data(uploaded_file) multiplier = df['Multiplier(Crash)']

st.subheader("📊 Crash Data Preview")
st.dataframe(df.tail(10))

st.subheader("⚙️ Select Models to Use")
selected_models = st.multiselect("Choose prediction models:", list(models.keys()), default=list(models.keys()))

if st.button("🔮 Predict Next Crash"):
    st.subheader("📈 Predictions")
    for model_name in selected_models:
        try:
            value, acc = models[model_name](multiplier)
            strategy = "✅ Play" if acc >= 75 else "⛔ Skip"
            st.markdown(f"**{model_name}** → Prediction: `{value:.2f}x`, Confidence: `{acc:.1f}%`, Strategy: {strategy}")
        except Exception as e:
            st.error(f"Error with {model_name}: {str(e)}")

# Graph
st.subheader("📉 Crash Trend History")
fig, ax = plt.subplots()
ax.plot(multiplier.values, label="Multiplier", color='blue')
ax.set_title("Crash Multiplier History")
ax.set_xlabel("Round")
ax.set_ylabel("Multiplier (x)")
ax.legend()
st.pyplot(fig)

else: st.warning("Please upload a CSV file with a column named 'Multiplier(Crash)'.")

