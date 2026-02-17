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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Page Config ---
st.set_page_config(
    page_title="NanoFlow | Density Predictor",
    page_icon="🧪",
    layout="wide"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def train_model(df):
    X = df.drop(columns=['Density (ρ)'])
    y = df['Density (ρ)']
    cat_cols = ['Nano Particle', 'Base Fluid']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)], 
        remainder='passthrough'
    )
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X, y)
    return model

# --- Header Section ---
col_logo, col_text = st.columns([1, 4])
with col_text:
    st.title("Nanofluid Density Predictive Analytics")
    st.markdown("""
    This interface utilizes a **Random Forest Machine Learning** model to predict the effective density 
    of hybrid nanofluids based on experimental parameters.
    """)

# --- Theoretical Background ---
with st.expander("📚 View Theoretical Formula"):
    st.write("The density of a hybrid nanofluid is typically calculated using the mixture rule:")
    st.latex(r"\rho_{hnf} = (1 - \phi)\rho_{bf} + \phi_1\rho_{np1} + \phi_2\rho_{np2}")
    st.info("**Note:** The ML model accounts for non-linear temperature effects that this simple formula may overlook.")

try:
    data = pd.read_csv('Density_Prediction_Dataset.csv')
    model = train_model(data)

    # --- Sidebar Inputs ---
    st.sidebar.header("⚙️ Configuration")
    
    nano_p = st.sidebar.selectbox("Hybrid Nano Particle", data['Nano Particle'].unique())
    base_f = st.sidebar.selectbox("Base Fluid Media", data['Base Fluid'].unique())
    
    st.sidebar.divider()
    
    temp = st.sidebar.select_slider("System Temperature (°C)", 
                                    options=sorted(data['Temperature (°C)'].unique()))
    
    vol_conc = st.sidebar.slider("Total Volume Concentration (ϕ)", 0.0, 0.2, 0.05, 0.001)
    
    with st.sidebar.expander("Advanced Material Properties"):
        rho_np1 = st.number_input("ρ of Particle 1", value=float(data['Density of Nano Particle 1 (ρnp)'].mean()))
        rho_np2 = st.number_input("ρ of Particle 2", value=float(data['Density of Nano Particle 2 (ρnp)'].mean()))
        rho_bf = st.number_input("ρ of Base Fluid", value=float(data['Density of Base Fluid (ρbf)'].mean()))
        mix1 = st.slider("Mixture Ratio: Particle 1 (%)", 0, 100, 50)

    # --- Data Preparation ---
    input_dict = {
        'Nano Particle': nano_p, 'Base Fluid': base_f, 'Temperature (°C)': temp,
        'Volume Concentration (ϕ)': vol_conc, 'Density of Nano Particle 1 (ρnp)': rho_np1,
        'Density of Nano Particle 2 (ρnp)': rho_np2, 'Density of Base Fluid (ρbf)': rho_bf,
        'Volume Mixture of Particle 1': mix1, 'Volume Mixture of Particle 2': 100 - mix1
    }
    input_df = pd.DataFrame([input_dict])

    # --- Results Section ---
    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        st.subheader("Analysis Results")
        prediction = model.predict(input_df)[0]
        
        # Display Prediction with visual context
        delta_rho = prediction - rho_bf
        st.metric(label="Predicted Hybrid Nanofluid Density (ρ)", 
                  value=f"{prediction:,.2f} kg/m³", 
                  delta=f"{delta_rho:.2f} vs Base Fluid")
        
        st.success(f"The addition of {nano_p} increased the density by {((prediction/rho_bf)-1)*100:.2f}% relative to the base fluid.")

    with res_col2:
        st.subheader("Material Composition")
        # Visualizing the mixture
        chart_data = pd.DataFrame({
            'Component': ['Particle 1', 'Particle 2'],
            'Ratio': [mix1, 100-mix1]
        })
        st.write(f"**Primary Particle:** {nano_p.split('/')[0]}")
        st.bar_chart(chart_data.set_index('Component'))

    st.divider()
    
    # --- Experimental Data Preview ---
    tab1, tab2 = st.tabs(["📊 Correlation Matrix", "📋 Experimental Records"])
    with tab1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Scanning_Electron_Micrograph_of_Nanoparticles.jpg/640px-Scanning_Electron_Micrograph_of_Nanoparticles.jpg", 
                 caption="Reference SEM image of standard Nanoparticles (Illustration)")
    with tab2:
        st.dataframe(data, use_container_width=True)

except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.info("Ensure 'Density_Prediction_Dataset.csv' is present in the root directory.")
    
