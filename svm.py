import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NanoFlow AI | Hybrid Nanofluid Analytics",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main { background-color: #f8faff; }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e1e8f0;
    }
    div[data-testid="stExpander"] {
        border: none !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #1e3a8a;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL TRAINING ENGINE ---
@st.cache_resource
def train_model(df):
    """Trains a Random Forest Regressor and calculates evaluation metrics."""
    X = df.drop(columns=['Density (ρ)'])
    y = df['Density (ρ)']
    
    # Split data to evaluate performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    cat_cols = ['Nano Particle', 'Base Fluid']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ], remainder='passthrough'
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # Fit on training set
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Refit on full data for best production prediction (optional, but standard)
    pipeline.fit(X, y)
    
    return pipeline, r2, mae

# --- HEADER SECTION ---
st.title("🧪 Nanofluid Density Predictive Analytics")
st.markdown("""
    **Advanced Research Tool:** This interface predicts the effective density ($\rho_{hnf}$) of hybrid nanofluids 
    using a Random Forest Ensemble model trained on experimental data.
    """)
st.divider()

# --- LOAD DATA ---
try:
    data = pd.read_csv('Density_Prediction_Dataset.csv')
    # Unpack model and metrics
    model, r2_val, mae_val = train_model(data)

    # --- SIDEBAR: CONTROL PANEL ---
    st.sidebar.header("🔬 Configuration Panel")
    
    with st.sidebar:
        st.subheader("Material Selection")
        nano_p = st.selectbox("Hybrid Nano Particle Pair", sorted(data['Nano Particle'].unique()))
        base_f = st.selectbox("Base Fluid Medium", sorted(data['Base Fluid'].unique()))
        
        st.divider()
        st.subheader("Physical Parameters")
        temp = st.slider("System Temperature (°C)", 
                         float(data['Temperature (°C)'].min()), 
                         float(data['Temperature (°C)'].max()), 25.0)
        
        vol_conc = st.number_input("Total Volume Concentration (ϕ)", 
                                    min_value=0.0, max_value=0.5, 
                                    value=0.05, step=0.001, format="%.4f")
        
        with st.expander("🛠️ Advanced Component Density"):
            rho_np1 = st.number_input("Density P1 (kg/m³)", value=float(data['Density of Nano Particle 1 (ρnp)'].mean()))
            rho_np2 = st.number_input("Density P2 (kg/m³)", value=float(data['Density of Nano Particle 2 (ρnp)'].mean()))
            rho_bf = st.number_input("Density BF (kg/m³)", value=float(data['Density of Base Fluid (ρbf)'].mean()))
            mix1 = st.slider("Mix Ratio Particle 1 (%)", 0, 100, 50)
            mix2 = 100 - mix1

    # --- PREDICTION ENGINE ---
    input_features = pd.DataFrame([{
        'Nano Particle': nano_p,
        'Base Fluid': base_f,
        'Temperature (°C)': temp,
        'Volume Concentration (ϕ)': vol_conc,
        'Density of Nano Particle 1 (ρnp)': rho_np1,
        'Density of Nano Particle 2 (ρnp)': rho_np2,
        'Density of Base Fluid (ρbf)': rho_bf,
        'Volume Mixture of Particle 1': mix1,
        'Volume Mixture of Particle 2': mix2
    }])

    prediction = model.predict(input_features)[0]

    # --- MAIN UI LAYOUT ---
    st.subheader("Results & Model Performance")
    
    # Layout for Results (Added metrics for R2 and MAE)
    res_col1, res_col2, res_col3, res_col4 = st.columns([1.5, 1, 1, 1])

    with res_col1:
        st.write("**Theoretical Context**")
        st.latex(r"\rho_{hnf} = (1 - \phi)\rho_{bf} + \phi_1\rho_{np1} + \phi_2\rho_{np2}")
        st.info(f"Relative change to base fluid: **{((prediction/rho_bf)-1)*100:.2f}%**")

    with res_col2:
        delta_val = prediction - rho_bf
        st.metric(
            label="Estimated Density (ρ)", 
            value=f"{prediction:.2f} kg/m³", 
            delta=f"{delta_val:.2f} kg/m³",
            delta_color="normal"
        )

    with res_col3:
        st.metric(
            label="Model $R^2$ Score", 
            value=f"{r2_val:.4f}",
            help="Closer to 1.0000 is perfect prediction."
        )

    with res_col4:
        st.metric(
            label="Mean Absolute Error", 
            value=f"{mae_val:.2f}",
            help="Average units (kg/m³) the prediction deviates from reality."
        )

    st.markdown("---")

    # --- ANALYTICS SECTION ---
    st.subheader("📊 Data Science & Insights")
    tab_corr, tab_trend, tab_data = st.tabs(["Correlation Heatmap", "Temperature Trends", "Dataset Explorer"])

    with tab_corr:
        st.write("### Pearson Correlation Matrix")
        numeric_df = data.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax_corr)
        plt.title("Property Correlation Heatmap", fontsize=14)
        st.pyplot(fig_corr)

    with tab_trend:
        st.write("### Temperature vs. Density Analysis")
        fig_trend, ax_trend = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=data, x='Temperature (°C)', y='Density (ρ)', hue='Nano Particle', marker='o', ax=ax_trend)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.title("Impact of Temperature on Density Across Particles")
        st.pyplot(fig_trend)

    with tab_data:
        st.write("### Experimental Training Data")
        st.dataframe(data, use_container_width=True)
        st.download_button("📥 Export Experimental Data to CSV", 
                            data.to_csv(index=False), 
                            file_name="Nanofluid_Density_Data.csv", 
                            mime="text/csv")

except Exception as e:
    st.error(f"❌ Application Error: {e}")
    st.info("Check if 'Density_Prediction_Dataset.csv' is in the same directory.")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown("© 2024 Nanofluid ML Analytics v2.0")
