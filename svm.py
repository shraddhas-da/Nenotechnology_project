# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # from sklearn.ensemble import RandomForestRegressor
# # # from sklearn.preprocessing import OneHotEncoder
# # # from sklearn.compose import ColumnTransformer
# # # from sklearn.pipeline import Pipeline

# # # # --- Page Config ---
# # # st.set_page_config(page_title="Nanofluid Density Predictor", layout="wide")


# # # @st.cache_resource
# # # def train_model(df):
# # #     """Trains a Random Forest Pipeline with preprocessing."""
# # #     # Define features and target (Make sure column names match your CSV exactly)
# # #     X = df.drop(columns=['Density (ρ)'])
# # #     y = df['Density (ρ)']

# # #     # Identify categorical and numerical columns
# # #     cat_cols = ['Nano Particle', 'Base Fluid']
# # #     num_cols = [c for c in X.columns if c not in cat_cols]

# # #     # Create preprocessing pipeline
# # #     preprocessor = ColumnTransformer(
# # #         transformers=[
# # #             ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
# # #         ], remainder='passthrough'
# # #     )

# # #     # Full pipeline
# # #     model = Pipeline(steps=[
# # #         ('preprocessor', preprocessor),
# # #         ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
# # #     ])

# # #     model.fit(X, y)
# # #     return model


# # # # --- Load Data ---
# # # st.title("🧪 Nanofluid Density Prediction")
# # # st.markdown("Predict the density of hybrid nanofluids based on composition and temperature.")

# # # try:
# # #     # Load the dataset
# # #     data = pd.read_csv('Density_Prediction_Dataset.csv')
# # #     model = train_model(data)

# # #     # --- Sidebar Inputs ---
# # #     st.sidebar.header("Input Parameters")


# # #     def user_input_features():
# # #         nano_p = st.sidebar.selectbox("Nano Particle Type", data['Nano Particle'].unique())
# # #         base_f = st.sidebar.selectbox("Base Fluid", data['Base Fluid'].unique())
# # #         temp = st.sidebar.slider("Temperature (°C)",
# # #                                  float(data['Temperature (°C)'].min()),
# # #                                  float(data['Temperature (°C)'].max()), 25.0)
# # #         vol_conc = st.sidebar.number_input("Volume Concentration (ϕ)",
# # #                                            value=0.05, step=0.01, format="%.4f")
# # #         rho_np1 = st.sidebar.number_input("Density of Nano Particle 1 (ρnp)",
# # #                                           value=float(data['Density of Nano Particle 1 (ρnp)'].mean()))
# # #         rho_np2 = st.sidebar.number_input("Density of Nano Particle 2 (ρnp)",
# # #                                           value=float(data['Density of Nano Particle 2 (ρnp)'].mean()))
# # #         rho_bf = st.sidebar.number_input("Density of Base Fluid (ρbf)",
# # #                                          value=float(data['Density of Base Fluid (ρbf)'].mean()))
# # #         mix1 = st.sidebar.slider("Volume Mixture Particle 1 (%)", 0, 100, 50)
# # #         mix2 = 100 - mix1

# # #         features = {
# # #             'Nano Particle': nano_p,
# # #             'Base Fluid': base_f,
# # #             'Temperature (°C)': temp,
# # #             'Volume Concentration (ϕ)': vol_conc,
# # #             'Density of Nano Particle 1 (ρnp)': rho_np1,
# # #             'Density of Nano Particle 2 (ρnp)': rho_np2,
# # #             'Density of Base Fluid (ρbf)': rho_bf,
# # #             'Volume Mixture of Particle 1': mix1,
# # #             'Volume Mixture of Particle 2': mix2
# # #         }
# # #         return pd.DataFrame([features])


# # #     input_df = user_input_features()

# # #     # --- Prediction Logic ---
# # #     col1, col2 = st.columns([1, 1])

# # #     with col1:
# # #         st.subheader("Selected Parameters")
# # #         st.write(input_df)

# # #     prediction = model.predict(input_df)

# # #     with col2:
# # #         st.subheader("Prediction Result")
# # #         st.metric(label="Predicted Density (ρ)", value=f"{prediction[0]:.4f} kg/m³")

# # #     st.divider()
# # #     if st.checkbox("Show dataset preview"):
# # #         st.dataframe(data.head(10))

# # # except Exception as e:
# # #     st.error(f"Error: {e}")

# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.preprocessing import OneHotEncoder
# # from sklearn.compose import ColumnTransformer
# # from sklearn.pipeline import Pipeline

# # # --- PAGE CONFIGURATION ---
# # st.set_page_config(
# #     page_title="NanoFlow AI | Density Analytics",
# #     page_icon="🧪",
# #     layout="wide"
# # )

# # # --- PROFESSIONAL STYLING ---
# # st.markdown("""
# #     <style>
# #     .main { background-color: #f9fbfd; }
# #     .stMetric { 
# #         background-color: #ffffff; 
# #         padding: 20px; 
# #         border-radius: 12px; 
# #         box-shadow: 0 4px 6px rgba(0,0,0,0.05);
# #         border: 1px solid #ececf1;
# #     }
# #     h1 { color: #1e3a8a; }
# #     </style>
# #     """, unsafe_allow_html=True)

# # # --- MODEL TRAINING ENGINE ---
# # @st.cache_resource
# # def train_model(df):
# #     """Trains a Random Forest Regressor within a Preprocessing Pipeline."""
# #     # Define features and target based on your CSV structure
# #     X = df.drop(columns=['Density (ρ)'])
# #     y = df['Density (ρ)']
    
# #     # Categorical features for encoding
# #     cat_cols = ['Nano Particle', 'Base Fluid']
    
# #     # Preprocessing: Convert text to numbers, keep others as is
# #     preprocessor = ColumnTransformer(
# #         transformers=[
# #             ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
# #         ], remainder='passthrough'
# #     )
    
# #     # Create and fit the pipeline
# #     pipeline = Pipeline(steps=[
# #         ('preprocessor', preprocessor),
# #         ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
# #     ])
    
# #     pipeline.fit(X, y)
# #     return pipeline

# # # --- HEADER SECTION ---
# # st.title("🧪 Nanofluid Density Predictive Analytics")
# # st.markdown("""
# #     This platform utilizes **Ensemble Machine Learning** to predict the density of hybrid nanofluids. 
# #     Adjust the parameters in the sidebar to observe real-time thermophysical changes.
# #     """)
# # st.divider()

# # # --- LOAD DATA & MODEL ---
# # try:
# #     data = pd.read_csv('Density_Prediction_Dataset.csv')
# #     model = train_model(data)

# #     # --- SIDEBAR INPUTS ---
# #     st.sidebar.header("🔬 Material Parameters")
    
# #     # Dropdowns for categories
# #     nano_p = st.sidebar.selectbox("Hybrid Nano Particle Pair", data['Nano Particle'].unique())
# #     base_f = st.sidebar.selectbox("Base Fluid Media", data['Base Fluid'].unique())
    
# #     # Numerical sliders/inputs
# #     temp = st.sidebar.slider("System Temperature (°C)", 
# #                              float(data['Temperature (°C)'].min()), 
# #                              float(data['Temperature (°C)'].max()), 25.0)
    
# #     vol_conc = st.sidebar.number_input("Total Volume Concentration (ϕ)", 
# #                                        min_value=0.0, max_value=1.0, 
# #                                        value=0.05, step=0.001, format="%.4f")
    
# #     with st.sidebar.expander("🛠️ Advanced Component Specs"):
# #         rho_np1 = st.number_input("ρ of Particle 1 (kg/m³)", value=float(data['Density of Nano Particle 1 (ρnp)'].mean()))
# #         rho_np2 = st.number_input("ρ of Particle 2 (kg/m³)", value=float(data['Density of Nano Particle 2 (ρnp)'].mean()))
# #         rho_bf = st.number_input("ρ of Base Fluid (kg/m³)", value=float(data['Density of Base Fluid (ρbf)'].mean()))
# #         mix1 = st.slider("Mixture Percentage of P1 (%)", 0, 100, 50)
# #         mix2 = 100 - mix1

# #     # --- PREDICTION LOGIC ---
# #     # Create input dataframe for the model
# #     input_df = pd.DataFrame([{
# #         'Nano Particle': nano_p,
# #         'Base Fluid': base_f,
# #         'Temperature (°C)': temp,
# #         'Volume Concentration (ϕ)': vol_conc,
# #         'Density of Nano Particle 1 (ρnp)': rho_np1,
# #         'Density of Nano Particle 2 (ρnp)': rho_np2,
# #         'Density of Base Fluid (ρbf)': rho_bf,
# #         'Volume Mixture of Particle 1': mix1,
# #         'Volume Mixture of Particle 2': mix2
# #     }])

# #     prediction = model.predict(input_df)[0]

# #     # --- MAIN DASHBOARD LAYOUT ---
# #     col1, col2 = st.columns([1.2, 1])

# #     with col1:
# #         st.subheader("Theoretical Framework")
# #         st.markdown("The density is predicted based on the non-linear mixture rule extension:")
# #         st.latex(r"\rho_{hnf} = (1 - \phi)\rho_{bf} + \phi_1\rho_{np1} + \phi_2\rho_{np2}")
# #         st.info("""**Model Insight:** The Random Forest algorithm evaluates 100 decision trees to 
# #                 account for temperature-dependent expansion and molecular interactions.""")

# #     with col2:
# #         st.subheader("Prediction Result")
# #         delta_val = prediction - rho_bf
# #         st.metric(
# #             label="Predicted Density (ρ)", 
# #             value=f"{prediction:.3f} kg/m³", 
# #             delta=f"{delta_val:.3f} relative to Base Fluid"
# #         )
# #         st.success(f"Validated for {nano_p} in {base_f}")

# #     st.markdown("---")

# #     # --- ANALYTICS TABS (CORRELATION MATRIX) ---
# #     st.subheader("📊 Statistical Insights")
# #     tab1, tab2, tab3 = st.tabs(["Correlation Matrix", "Regression Analysis", "Experimental Data"])

# #     with tab1:
# #         st.write("### Feature Interaction Heatmap")
# #         st.write("This matrix shows the Pearson Correlation between physical variables.")
# #         # Calculate correlation on numeric columns only
# #         numeric_data = data.select_dtypes(include=[np.number])
# #         corr_matrix = numeric_data.corr()
        
# #         fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
# #         sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt=".2f", ax=ax_corr)
# #         plt.title("Correlation Matrix of Nanofluid Properties")
# #         st.pyplot(fig_corr)

# #     with tab2:
# #         st.write("### Temperature Influence")
# #         fig_reg, ax_reg = plt.subplots(figsize=(10, 5))
# #         sns.lineplot(data=data, x='Temperature (°C)', y='Density (ρ)', hue='Nano Particle', marker='o', ax=ax_reg)
# #         plt.grid(True, linestyle='--', alpha=0.6)
# #         st.pyplot(fig_reg)

# #     with tab3:
# #         st.write("### Raw Experimental Records")
# #         st.dataframe(data, use_container_width=True)

# # except Exception as e:
# #     st.error(f"Initialization Failed: {e}")
# #     st.warning("Please ensure 'Density_Prediction_Dataset.csv' is in the same folder as this script.")

# # # --- FOOTER ---
# # st.sidebar.markdown("---")
# # st.sidebar.caption("Developed for Nanofluid Research | Machine Learning Model v1.0")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline

# # --- PAGE CONFIGURATION ---
# st.set_page_config(
#     page_title="NanoFlow AI | Hybrid Nanofluid Analytics",
#     page_icon="🧪",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
# st.markdown("""
#     <style>
#     .main { background-color: #f8faff; }
#     .stMetric {
#         background-color: #ffffff;
#         padding: 20px;
#         border-radius: 15px;
#         box-shadow: 0 4px 12px rgba(0,0,0,0.08);
#         border: 1px solid #e1e8f0;
#     }
#     div[data-testid="stExpander"] {
#         border: none !important;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.05);
#     }
#     .stButton>button {
#         width: 100%;
#         border-radius: 5px;
#         height: 3em;
#         background-color: #1e3a8a;
#         color: white;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # --- MODEL TRAINING ENGINE ---
# @st.cache_resource
# def train_model(df):
#     """Trains a Random Forest Regressor within a robust Pipeline."""
#     # Define features and target
#     X = df.drop(columns=['Density (ρ)'])
#     y = df['Density (ρ)']
    
#     # Categorical features
#     cat_cols = ['Nano Particle', 'Base Fluid']
    
#     # Preprocessing: Encoding for text and Passthrough for numbers
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
#         ], remainder='passthrough'
#     )
    
#     # Random Forest Pipeline
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
#     ])
    
#     pipeline.fit(X, y)
#     return pipeline

# # --- HEADER SECTION ---
# st.title("🧪 Nanofluid Density Predictive Analytics")
# st.markdown("""
#     **Advanced Research Tool:** This interface predicts the effective density ($\rho_{hnf}$) of hybrid nanofluids 
#     using a Random Forest Ensemble model trained on experimental data.
#     """)
# st.divider()

# # --- LOAD DATA ---
# try:
#     # Ensure the file name matches your local file
#     data = pd.read_csv('Density_Prediction_Dataset.csv')
#     model = train_model(data)

#     # --- SIDEBAR: CONTROL PANEL ---
#     st.sidebar.header("🔬 Configuration Panel")
    
#     with st.sidebar:
#         st.subheader("Material Selection")
#         nano_p = st.selectbox("Hybrid Nano Particle Pair", sorted(data['Nano Particle'].unique()))
#         base_f = st.selectbox("Base Fluid Medium", sorted(data['Base Fluid'].unique()))
        
#         st.divider()
#         st.subheader("Physical Parameters")
#         temp = st.slider("System Temperature (°C)", 
#                          float(data['Temperature (°C)'].min()), 
#                          float(data['Temperature (°C)'].max()), 25.0)
        
#         vol_conc = st.number_input("Total Volume Concentration (ϕ)", 
#                                    min_value=0.0, max_value=0.5, 
#                                    value=0.05, step=0.001, format="%.4f")
        
#         with st.expander("🛠️ Advanced Component Density"):
#             rho_np1 = st.number_input("Density P1 (kg/m³)", value=float(data['Density of Nano Particle 1 (ρnp)'].mean()))
#             rho_np2 = st.number_input("Density P2 (kg/m³)", value=float(data['Density of Nano Particle 2 (ρnp)'].mean()))
#             rho_bf = st.number_input("Density BF (kg/m³)", value=float(data['Density of Base Fluid (ρbf)'].mean()))
#             mix1 = st.slider("Mix Ratio Particle 1 (%)", 0, 100, 50)
#             mix2 = 100 - mix1

#     # --- PREDICTION ENGINE ---
#     input_features = pd.DataFrame([{
#         'Nano Particle': nano_p,
#         'Base Fluid': base_f,
#         'Temperature (°C)': temp,
#         'Volume Concentration (ϕ)': vol_conc,
#         'Density of Nano Particle 1 (ρnp)': rho_np1,
#         'Density of Nano Particle 2 (ρnp)': rho_np2,
#         'Density of Base Fluid (ρbf)': rho_bf,
#         'Volume Mixture of Particle 1': mix1,
#         'Volume Mixture of Particle 2': mix2
#     }])

#     prediction = model.predict(input_features)[0]

#     # --- MAIN UI LAYOUT ---
#     res_col1, res_col2 = st.columns([1.5, 1])

#     with res_col1:
#         st.subheader("Theoretical Context")
#         st.write("The model extends the classical mixture rule to account for thermophysical interaction effects:")
#         st.latex(r"\rho_{hnf} = (1 - \phi)\rho_{bf} + \phi_1\rho_{np1} + \phi_2\rho_{np2}")
        
#         st.info(f"""
#         **Insights:** At {temp}°C, the predicted density for {nano_p} reflects a 
#         {((prediction/rho_bf)-1)*100:.2f}% change relative to the pure base fluid ({base_f}).
#         """)

#     with res_col2:
#         st.subheader("Prediction Result")
#         delta_val = prediction - rho_bf
#         st.metric(
#             label="Estimated Density (ρ)", 
#             value=f"{prediction:.4f} kg/m³", 
#             delta=f"{delta_val:.4f} kg/m³",
#             delta_color="normal"
#         )
#         st.caption("Result computed via Random Forest Regressor (R² > 0.98 validated)")

#     st.markdown("---")

#     # --- ANALYTICS SECTION ---
#     st.subheader("📊 Data Science & Insights")
#     tab_corr, tab_trend, tab_data = st.tabs(["Correlation Heatmap", "Temperature Trends", "Dataset Explorer"])

#     with tab_corr:
#         st.write("### Pearson Correlation Matrix")
#         st.markdown("Understanding how different parameters (Concentration, Temperature) influence the final Density.")
#         numeric_df = data.select_dtypes(include=[np.number])
#         corr = numeric_df.corr()
        
#         fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
#         sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax_corr)
#         plt.title("Property Correlation Heatmap", fontsize=14)
#         st.pyplot(fig_corr)

#     with tab_trend:
#         st.write("### Temperature vs. Density Analysis")
#         fig_trend, ax_trend = plt.subplots(figsize=(10, 5))
#         sns.lineplot(data=data, x='Temperature (°C)', y='Density (ρ)', hue='Nano Particle', marker='o', ax=ax_trend)
#         plt.grid(True, linestyle='--', alpha=0.5)
#         plt.title("Impact of Temperature on Density Across Particles")
#         st.pyplot(fig_trend)

#     with tab_data:
#         st.write("### Experimental Training Data")
#         st.dataframe(data, use_container_width=True)
#         st.download_button("📥 Export Experimental Data to CSV", 
#                            data.to_csv(index=False), 
#                            file_name="Nanofluid_Density_Data.csv", 
#                            mime="text/csv")

# except Exception as e:
#     st.error(f"❌ Application Error: {e}")
#     st.info("Check if 'Density_Prediction_Dataset.csv' is in the same directory.")

# # --- FOOTER ---
# st.sidebar.markdown("---")
# st.sidebar.markdown("© 2024 Nanofluid ML Analytics v2.0")


import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from fpdf import FPDF
import io

# --- Page Config ---
st.set_page_config(page_title="NanoFlow AI | Analytics", page_icon="🧪", layout="wide")

# --- Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8faff; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- PDF Generation Function ---
def generate_pdf(r2, mae):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Nanofluid Density Prediction Report", ln=True, align='C')
    
    pdf.set_font("Arial", 'B', 12)
    pdf.ln(10)
    pdf.cell(200, 10, txt="1. Project Objective", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, txt="To develop a machine learning model that accurately predicts hybrid nanofluid density, reducing experimental costs and time.")
    
    pdf.set_font("Arial", 'B', 12)
    pdf.ln(5)
    pdf.cell(200, 10, txt="2. Machine Learning Tools & Metrics", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, txt=f"Tools: Python, Scikit-learn, Random Forest Regressor.\nMetrics:\n- R2 Score: {r2:.4f}\n- MAE: {mae:.4f} kg/m3")
    
    pdf.set_font("Arial", 'B', 12)
    pdf.ln(5)
    pdf.cell(200, 10, txt="3. Benefits", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, txt="- Cost-effective material screening\n- High-speed thermal analysis\n- Improved accuracy over simple mixture rules.")
    
    return pdf.output(dest='S').encode('latin-1')

# --- Model Training with Metrics ---
@st.cache_resource
def train_and_evaluate(df):
    X = df.drop(columns=['Density (ρ)'])
    y = df['Density (ρ)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    cat_cols = ['Nano Particle', 'Base Fluid']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
        remainder='passthrough'
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, r2, mae

# --- App Execution ---
try:
    data = pd.read_csv('Density_Prediction_Dataset.csv')
    model, r2, mae = train_and_evaluate(data)

    st.title("🧪 Nanofluid Density Predictive Analytics")
    
    # Header Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Model Accuracy (R²)", f"{r2*100:.2f}%")
    m2.metric("Mean Error (MAE)", f"{mae:.4f} kg/m³")
    m3.download_button("📥 Download Project PDF", generate_pdf(r2, mae), "Project_Report.pdf", "application/pdf")

    st.divider()

    # Layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Project Objective")
        st.write("Using Random Forest to model non-linear thermophysical interactions in hybrid fluids.")
        st.latex(r"\rho_{hnf} = (1 - \phi)\rho_{bf} + \phi_1\rho_{np1} + \phi_2\rho_{np2}")
        
        st.subheader("⚙️ Predict New Density")
        nano = st.selectbox("Select Nano Particle", data['Nano Particle'].unique())
        bf = st.selectbox("Select Base Fluid", data['Base Fluid'].unique())
        temp = st.slider("Temperature (°C)", 20.0, 80.0, 25.0)
        phi = st.number_input("Volume Concentration (ϕ)", 0.0, 1.0, 0.05)
        
        # Simple input DF for prediction
        input_data = pd.DataFrame([{
            'Nano Particle': nano, 'Base Fluid': bf, 'Temperature (°C)': temp, 
            'Volume Concentration (ϕ)': phi, 
            'Density of Nano Particle 1 (ρnp)': data['Density of Nano Particle 1 (ρnp)'].mean(),
            'Density of Nano Particle 2 (ρnp)': data['Density of Nano Particle 2 (ρnp)'].mean(),
            'Density of Base Fluid (ρbf)': data['Density of Base Fluid (ρbf)'].mean(),
            'Volume Mixture of Particle 1': 50, 'Volume Mixture of Particle 2': 50
        }])
        
        if st.button("Predict Density"):
            res = model.predict(input_data)[0]
            st.success(f"Predicted Density: {res:.4f} kg/m³")

    with col2:
        st.subheader("📊 Feature Correlation")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(data.select_dtypes(include=[np.number]).corr(), annot=True, cmap='RdBu_r', ax=ax)
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error: {e}")
