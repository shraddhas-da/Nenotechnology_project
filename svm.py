import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Page Config ---
st.set_page_config(page_title="Nanofluid Density Predictor", layout="wide")


@st.cache_resource
def train_model(df):
    """Trains a Random Forest Pipeline with preprocessing."""
    # Define features and target (Make sure column names match your CSV exactly)
    X = df.drop(columns=['Density (ρ)'])
    y = df['Density (ρ)']

    # Identify categorical and numerical columns
    cat_cols = ['Nano Particle', 'Base Fluid']
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ], remainder='passthrough'
    )

    # Full pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)
    return model


# --- Load Data ---
st.title("🧪 Nanofluid Density Prediction")
st.markdown("Predict the density of hybrid nanofluids based on composition and temperature.")

try:
    # Load the dataset
    data = pd.read_csv('Density_Prediction_Dataset.csv')
    model = train_model(data)

    # --- Sidebar Inputs ---
    st.sidebar.header("Input Parameters")


    def user_input_features():
        nano_p = st.sidebar.selectbox("Nano Particle Type", data['Nano Particle'].unique())
        base_f = st.sidebar.selectbox("Base Fluid", data['Base Fluid'].unique())
        temp = st.sidebar.slider("Temperature (°C)",
                                 float(data['Temperature (°C)'].min()),
                                 float(data['Temperature (°C)'].max()), 25.0)
        vol_conc = st.sidebar.number_input("Volume Concentration (ϕ)",
                                           value=0.05, step=0.01, format="%.4f")
        rho_np1 = st.sidebar.number_input("Density of Nano Particle 1 (ρnp)",
                                          value=float(data['Density of Nano Particle 1 (ρnp)'].mean()))
        rho_np2 = st.sidebar.number_input("Density of Nano Particle 2 (ρnp)",
                                          value=float(data['Density of Nano Particle 2 (ρnp)'].mean()))
        rho_bf = st.sidebar.number_input("Density of Base Fluid (ρbf)",
                                         value=float(data['Density of Base Fluid (ρbf)'].mean()))
        mix1 = st.sidebar.slider("Volume Mixture Particle 1 (%)", 0, 100, 50)
        mix2 = 100 - mix1

        features = {
            'Nano Particle': nano_p,
            'Base Fluid': base_f,
            'Temperature (°C)': temp,
            'Volume Concentration (ϕ)': vol_conc,
            'Density of Nano Particle 1 (ρnp)': rho_np1,
            'Density of Nano Particle 2 (ρnp)': rho_np2,
            'Density of Base Fluid (ρbf)': rho_bf,
            'Volume Mixture of Particle 1': mix1,
            'Volume Mixture of Particle 2': mix2
        }
        return pd.DataFrame([features])


    input_df = user_input_features()

    # --- Prediction Logic ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Selected Parameters")
        st.write(input_df)

    prediction = model.predict(input_df)

    with col2:
        st.subheader("Prediction Result")
        st.metric(label="Predicted Density (ρ)", value=f"{prediction[0]:.4f} kg/m³")

    st.divider()
    if st.checkbox("Show dataset preview"):
        st.dataframe(data.head(10))

except Exception as e:
    st.error(f"Error: {e}")
