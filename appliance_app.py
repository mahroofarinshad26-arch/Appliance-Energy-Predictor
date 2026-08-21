import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Appliance Energy Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title-box {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.title-box h1 {
    font-size: 38px;
    margin-bottom: 5px;
}

.title-box p {
    font-size: 17px;
    margin-top: 5px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

.prediction-box {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white;
    text-align: center;
    margin-top: 25px;
}

.prediction-value {
    font-size: 38px;
    font-weight: bold;
}

.info-box {
    padding: 18px;
    border-radius: 12px;
    background-color: #f0f2f6;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD PIPELINE
# ============================================================

@st.cache_resource
def load_pipeline():
    return joblib.load("appliance_energy_pipeline.pkl")


pipeline = load_pipeline()

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="title-box">
    <h1> Appliance Energy Predictor</h1>
    <p>Machine Learning Based Household Energy Consumption Prediction</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Enter the environmental and household conditions below "
    "to estimate appliance energy consumption."
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(" About the Project")

    st.write(
        "This application uses a trained Machine Learning "
        "Regression Pipeline to predict household appliance "
        "energy consumption."
    )

    st.divider()

    st.subheader("Model Information")

    st.write(" Model: Random Forest Regressor")
    st.write(" Problem: Regression")
    st.write(" Target: Appliances")
    st.write(" Unit: Watt-hour (Wh)")

    st.divider()

    st.info(
        "The application sends raw user inputs directly "
        "to the saved ML pipeline."
    )

# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title"> Household Conditions</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.subheader(" Lighting")

    lights = st.number_input(
        "Lights",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0
    )

with col2:

    st.subheader(" Indoor Temperature")

    T1 = st.number_input("T1 - Kitchen Temperature (°C)", value=20.0)
    T2 = st.number_input("T2 - Living Room Temperature (°C)", value=20.0)
    T3 = st.number_input("T3 - Laundry Room Temperature (°C)", value=20.0)

with col3:

    st.subheader(" Indoor Humidity")

    RH_1 = st.number_input("RH_1 - Kitchen Humidity (%)", value=40.0)
    RH_2 = st.number_input("RH_2 - Living Room Humidity (%)", value=40.0)
    RH_3 = st.number_input("RH_3 - Laundry Room Humidity (%)", value=40.0)

# ============================================================
# MORE INDOOR CONDITIONS
# ============================================================

st.markdown(
    '<div class="section-title">🏠 Additional Indoor Conditions</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    T4 = st.number_input("T4 - Office Temperature (°C)", value=20.0)
    RH_4 = st.number_input("RH_4 - Office Humidity (%)", value=40.0)

    T5 = st.number_input("T5 - Bathroom Temperature (°C)", value=20.0)
    RH_5 = st.number_input("RH_5 - Bathroom Humidity (%)", value=40.0)

with col2:

    T6 = st.number_input("T6 - Outside North Temperature (°C)", value=10.0)
    RH_6 = st.number_input("RH_6 - Outside North Humidity (%)", value=70.0)

    T7 = st.number_input("T7 - Ironing Room Temperature (°C)", value=20.0)
    RH_7 = st.number_input("RH_7 - Ironing Room Humidity (%)", value=40.0)

with col3:

    T8 = st.number_input("T8 - Teenager Room Temperature (°C)", value=20.0)
    RH_8 = st.number_input("RH_8 - Teenager Room Humidity (%)", value=40.0)

    T9 = st.number_input("T9 - Parents Room Temperature (°C)", value=20.0)
    RH_9 = st.number_input("RH_9 - Parents Room Humidity (%)", value=40.0)

# ============================================================
# OUTDOOR CONDITIONS
# ============================================================

st.markdown(
    '<div class="section-title"> Outdoor & Weather Conditions</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    T_out = st.number_input(
        "Outside Temperature (°C)",
        value=10.0
    )

    RH_out = st.number_input(
        "Outside Humidity (%)",
        value=70.0
    )

with col2:

    Press_mm_hg = st.number_input(
        "Atmospheric Pressure (mm Hg)",
        value=750.0
    )

    Windspeed = st.number_input(
        "Wind Speed",
        min_value=0.0,
        value=4.0
    )

with col3:

    Visibility = st.number_input(
        "Visibility",
        min_value=0.0,
        value=40.0
    )

    Tdewpoint = st.number_input(
        "Dew Point Temperature (°C)",
        value=5.0
    )

# ============================================================
# RANDOM FEATURES
# ============================================================

st.markdown(
    '<div class="section-title"> Additional Features</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    rv1 = st.number_input(
        "Random Variable 1",
        value=0.0
    )

with col2:

    rv2 = st.number_input(
        "Random Variable 2",
        value=0.0
    )

# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "  Predict Energy Consumption",
    use_container_width=True,
    type="primary"
)

# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    input_data = pd.DataFrame([{

        "lights": lights,

        "T1": T1,
        "RH_1": RH_1,

        "T2": T2,
        "RH_2": RH_2,

        "T3": T3,
        "RH_3": RH_3,

        "T4": T4,
        "RH_4": RH_4,

        "T5": T5,
        "RH_5": RH_5,

        "T6": T6,
        "RH_6": RH_6,

        "T7": T7,
        "RH_7": RH_7,

        "T8": T8,
        "RH_8": RH_8,

        "T9": T9,
        "RH_9": RH_9,

        "T_out": T_out,
        "Press_mm_hg": Press_mm_hg,
        "RH_out": RH_out,
        "Windspeed": Windspeed,
        "Visibility": Visibility,
        "Tdewpoint": Tdewpoint,

        "rv1": rv1,
        "rv2": rv2
    }])

    # Prediction using complete pipeline
    prediction = pipeline.predict(input_data)[0]

    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(f"""
    <div class="prediction-box">
        <h2> Predicted Appliance Energy Consumption</h2>
        <div class="prediction-value">
            {prediction:.2f} Wh
        </div>
        <p>Estimated household appliance energy consumption</p>
    </div>
    """, unsafe_allow_html=True)

    st.success(
        "Prediction generated successfully using the trained "
        "Random Forest Regression Pipeline."
    )

    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    with st.expander(" View Input Summary"):

        st.dataframe(
            input_data,
            use_container_width=True
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚡ Appliance Energy Consumption Prediction | "
    "Machine Learning Regression Project"
)