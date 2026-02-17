# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline

# # --- Page Config ---
# st.set_page_config(page_title="Nanofluid Density Predictor", layout="wide")


# @st.cache_resource
# def train_model(df):
#     """Trains a Random Forest Pipeline with preprocessing."""
#     # Define features and target (Make sure column names match your CSV exactly)
#     X = df.drop(columns=['Density (ρ)'])
#     y = df['Density (ρ)']

#     # Identify categorical and numerical columns
#     cat_cols = ['Nano Particle', 'Base Fluid']
#     num_cols = [c for c in X.columns if c not in cat_cols]

#     # Create preprocessing pipeline
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
#         ], remainder='passthrough'
#     )

#     # Full pipeline
#     model = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
#     ])

#     model.fit(X, y)
#     return model


# # --- Load Data ---
# st.title("🧪 Nanofluid Density Prediction")
# st.markdown("Predict the density of hybrid nanofluids based on composition and temperature.")

# try:
#     # Load the dataset
#     data = pd.read_csv('Density_Prediction_Dataset.csv')
#     model = train_model(data)

#     # --- Sidebar Inputs ---
#     st.sidebar.header("Input Parameters")


#     def user_input_features():
#         nano_p = st.sidebar.selectbox("Nano Particle Type", data['Nano Particle'].unique())
#         base_f = st.sidebar.selectbox("Base Fluid", data['Base Fluid'].unique())
#         temp = st.sidebar.slider("Temperature (°C)",
#                                  float(data['Temperature (°C)'].min()),
#                                  float(data['Temperature (°C)'].max()), 25.0)
#         vol_conc = st.sidebar.number_input("Volume Concentration (ϕ)",
#                                            value=0.05, step=0.01, format="%.4f")
#         rho_np1 = st.sidebar.number_input("Density of Nano Particle 1 (ρnp)",
#                                           value=float(data['Density of Nano Particle 1 (ρnp)'].mean()))
#         rho_np2 = st.sidebar.number_input("Density of Nano Particle 2 (ρnp)",
#                                           value=float(data['Density of Nano Particle 2 (ρnp)'].mean()))
#         rho_bf = st.sidebar.number_input("Density of Base Fluid (ρbf)",
#                                          value=float(data['Density of Base Fluid (ρbf)'].mean()))
#         mix1 = st.sidebar.slider("Volume Mixture Particle 1 (%)", 0, 100, 50)
#         mix2 = 100 - mix1

#         features = {
#             'Nano Particle': nano_p,
#             'Base Fluid': base_f,
#             'Temperature (°C)': temp,
#             'Volume Concentration (ϕ)': vol_conc,
#             'Density of Nano Particle 1 (ρnp)': rho_np1,
#             'Density of Nano Particle 2 (ρnp)': rho_np2,
#             'Density of Base Fluid (ρbf)': rho_bf,
#             'Volume Mixture of Particle 1': mix1,
#             'Volume Mixture of Particle 2': mix2
#         }
#         return pd.DataFrame([features])


#     input_df = user_input_features()

#     # --- Prediction Logic ---
#     col1, col2 = st.columns([1, 1])

#     with col1:
#         st.subheader("Selected Parameters")
#         st.write(input_df)

#     prediction = model.predict(input_df)

#     with col2:
#         st.subheader("Prediction Result")
#         st.metric(label="Predicted Density (ρ)", value=f"{prediction[0]:.4f} kg/m³")

#     st.divider()
#     if st.checkbox("Show dataset preview"):
#         st.dataframe(data.head(10))

# except Exception as e:
#     st.error(f"Error: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NanoFlow AI | Density Analytics",
    page_icon="🧪",
    layout="wide"
)

# --- PROFESSIONAL STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f9fbfd; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #ececf1;
    }
    h1 { color: #1e3a8a; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL TRAINING ENGINE ---
@st.cache_resource
def train_model(df):
    """Trains a Random Forest Regressor within a Preprocessing Pipeline."""
    # Define features and target based on your CSV structure
    X = df.drop(columns=['Density (ρ)'])
    y = df['Density (ρ)']
    
    # Categorical features for encoding
    cat_cols = ['Nano Particle', 'Base Fluid']
    
    # Preprocessing: Convert text to numbers, keep others as is
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ], remainder='passthrough'
    )
    
    # Create and fit the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X, y)
    return pipeline

# --- HEADER SECTION ---
st.title("🧪 Nanofluid Density Predictive Analytics")
st.markdown("""
    This platform utilizes **Ensemble Machine Learning** to predict the density of hybrid nanofluids. 
    Adjust the parameters in the sidebar to observe real-time thermophysical changes.
    """)
st.divider()

# --- LOAD DATA & MODEL ---
try:
    data = pd.read_csv('Density_Prediction_Dataset.csv')
    model = train_model(data)

    # --- SIDEBAR INPUTS ---
    st.sidebar.header("🔬 Material Parameters")
    
    # Dropdowns for categories
    nano_p = st.sidebar.selectbox("Hybrid Nano Particle Pair", data['Nano Particle'].unique())
    base_f = st.sidebar.selectbox("Base Fluid Media", data['Base Fluid'].unique())
    
    # Numerical sliders/inputs
    temp = st.sidebar.slider("System Temperature (°C)", 
                             float(data['Temperature (°C)'].min()), 
                             float(data['Temperature (°C)'].max()), 25.0)
    
    vol_conc = st.sidebar.number_input("Total Volume Concentration (ϕ)", 
                                       min_value=0.0, max_value=1.0, 
                                       value=0.05, step=0.001, format="%.4f")
    
    with st.sidebar.expander("🛠️ Advanced Component Specs"):
        rho_np1 = st.number_input("ρ of Particle 1 (kg/m³)", value=float(data['Density of Nano Particle 1 (ρnp)'].mean()))
        rho_np2 = st.number_input("ρ of Particle 2 (kg/m³)", value=float(data['Density of Nano Particle 2 (ρnp)'].mean()))
        rho_bf = st.number_input("ρ of Base Fluid (kg/m³)", value=float(data['Density of Base Fluid (ρbf)'].mean()))
        mix1 = st.slider("Mixture Percentage of P1 (%)", 0, 100, 50)
        mix2 = 100 - mix1

    # --- PREDICTION LOGIC ---
    # Create input dataframe for the model
    input_df = pd.DataFrame([{
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

    prediction = model.predict(input_df)[0]

    # --- MAIN DASHBOARD LAYOUT ---
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Theoretical Framework")
        st.markdown("The density is predicted based on the non-linear mixture rule extension:")
        st.latex(r"\rho_{hnf} = (1 - \phi)\rho_{bf} + \phi_1\rho_{np1} + \phi_2\rho_{np2}")
        st.info("""**Model Insight:** The Random Forest algorithm evaluates 100 decision trees to 
                account for temperature-dependent expansion and molecular interactions.""")

    with col2:
        st.subheader("Prediction Result")
        delta_val = prediction - rho_bf
        st.metric(
            label="Predicted Density (ρ)", 
            value=f"{prediction:.3f} kg/m³", 
            delta=f"{delta_val:.3f} relative to Base Fluid"
        )
        st.success(f"Validated for {nano_p} in {base_f}")

    st.markdown("---")

    # --- ANALYTICS TABS (CORRELATION MATRIX) ---
    st.subheader("📊 Statistical Insights")
    tab1, tab2, tab3 = st.tabs(["Correlation Matrix", "Regression Analysis", "Experimental Data"])

    with tab1:
        st.write("### Feature Interaction Heatmap")
        st.write("This matrix shows the Pearson Correlation between physical variables.")
        # Calculate correlation on numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt=".2f", ax=ax_corr)
        plt.title("Correlation Matrix of Nanofluid Properties")
        st.pyplot(fig_corr)

    with tab2:
        st.write("### Temperature Influence")
        fig_reg, ax_reg = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=data, x='Temperature (°C)', y='Density (ρ)', hue='Nano Particle', marker='o', ax=ax_reg)
        plt.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_reg)

    with tab3:
        st.write("### Raw Experimental Records")
        st.dataframe(data, use_container_width=True)

except Exception as e:
    st.error(f"Initialization Failed: {e}")
    st.warning("Please ensure 'Density_Prediction_Dataset.csv' is in the same folder as this script.")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("Developed for Nanofluid Research | Machine Learning Model v1.0")
