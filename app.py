
import streamlit as st
import joblib
import numpy as np

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Healthcare AI | Clinical Analytics",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_breast_model():
    return joblib.load("breast_cancer_model.pkl")

@st.cache_resource
def load_insurance_model():
    return joblib.load("insurance_model.pkl")


# =====================================================
# PROFESSIONAL UI STYLING
# =====================================================

st.markdown("""
<style>

/* Main application */

.stApp {
    background-color: #F4F7FB;
    color: #1E293B;
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    max-width: 1450px;
    padding: 2rem 3rem;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #14243A;
    border-right: 1px solid #263B55;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF;
}

[data-testid="stSidebar"] hr {
    border-color: #334155;
}

/* Main header */

.header-container {
    background: linear-gradient(120deg, #14243A, #1E4E83);
    padding: 38px 35px;
    border-radius: 16px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 8px 24px rgba(15, 35, 60, 0.12);
}

.header-label {
    color: #93C5FD;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.header-container h1 {
    color: white !important;
    font-size: 34px;
    font-weight: 750;
    margin: 12px 0;
}

.header-container p {
    color: #DCEAFE;
    font-size: 15px;
    margin-bottom: 0;
}

/* Section headings */

.section-title {
    color: #14243A;
    font-size: 25px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 8px;
}

.section-description {
    color: #64748B;
    font-size: 14px;
    margin-bottom: 22px;
}

/* Metric cards */

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
}

[data-testid="stMetricLabel"] {
    color: #64748B;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #1E4E83;
    font-weight: 750;
}

/* Form containers */

[data-testid="stForm"] {
    background: #FFFFFF;
    padding: 24px;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
}

/* Input fields */

.stTextInput input,
.stNumberInput input {
    background-color: #F8FAFC;
    border-radius: 8px;
    border: 1px solid #CBD5E1;
    color: #1E293B;
}

.stSelectbox div[data-baseweb="select"] {
    background-color: #F8FAFC;
    border-radius: 8px;
}

/* Buttons */

.stButton > button,
.stFormSubmitButton > button {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 650;
    transition: 0.2s;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #1D4ED8;
    color: white;
    border: none;
}

/* Result panels */

.result-panel {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 5px solid #2563EB;
    border-radius: 12px;
    padding: 24px;
    margin: 18px 0;
}

.result-panel h3 {
    color: #14243A;
    margin-bottom: 10px;
}

.result-panel p {
    color: #64748B;
    margin-bottom: 5px;
}

.benign {
    border-left-color: #16A34A;
}

.malignant {
    border-left-color: #DC2626;
}

/* Footer */

.footer {
    text-align: center;
    color: #64748B;
    font-size: 12px;
    padding: 25px 0;
    border-top: 1px solid #E2E8F0;
    margin-top: 45px;
}

/* Download button */

.stDownloadButton > button {
    background: #14243A;
    color: white;
    border-radius: 8px;
    border: none;
}

.stDownloadButton > button:hover {
    background: #1E4E83;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================


with st.sidebar:

    st.markdown("""
<div style="padding: 15px 0 25px 0;">
    <div style="font-size:12px; letter-spacing:2px; color:#93C5FD; font-weight:700;">
        HEALTHCARE AI
    </div>
    <h2 style="color:white; font-size:25px; margin-top:8px;">
        Clinical Analytics
    </h2>
    <p style="color:#94A3B8; font-size:13px;">
        Intelligent healthcare prediction platform
    </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### WORKSPACE")

    option = st.radio(
        "Select Module",
        ["Breast Cancer Prediction", "Insurance Spend Prediction"],
        label_visibility="collapsed"
    )


    st.markdown("---")

    st.markdown("### WORKSPACE")

    option = st.radio(
    "Select Module",
    ["Breast Cancer Prediction", "Insurance Spend Prediction"],
    key="healthcare_module_navigation"
)

    st.markdown("---")

    st.markdown("### MODEL PERFORMANCE")

    st.metric(
        "Breast Cancer Accuracy",
        "95.61%"
    )

    st.metric(
        "Insurance R² Score",
        "91.77%"
    )

    st.markdown("---")

    st.markdown("### PROJECT INFORMATION")

    st.markdown("""
    **Developer**

    Nikhil Nathan

    **Machine Learning Models**

    - Random Forest Classifier
    - Random Forest Regressor

    **Framework**

    Streamlit

    **Version**

    1.0
    """)


# =====================================================
# MAIN HEADER
# =====================================================


st.markdown("""
<div class="header-container">
    <div class="header-label">
        HEALTHCARE INTELLIGENCE PLATFORM
    </div>
    <h1>Healthcare AI Platform</h1>
    <p>
        Machine Learning Powered Clinical Decision Support System
    </p>
    <p style="margin-top:18px; font-size:13px;">
        Breast Cancer Prediction &nbsp; | &nbsp;
        Insurance Spend Forecasting &nbsp; | &nbsp;
        Predictive Analytics
    </p>
</div>
""", unsafe_allow_html=True)



# =====================================================
# MODEL OVERVIEW
# =====================================================

st.markdown("### PLATFORM OVERVIEW")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "Prediction Modules",
        "02",
        "Machine Learning"
    )

with m2:
    st.metric(
        "Breast Cancer Accuracy",
        "95.61%",
        "Model Performance"
    )

with m3:
    st.metric(
        "Insurance R² Score",
        "91.77%",
        "Regression Performance"
    )

st.markdown("---")


# =====================================================
# BREAST CANCER PREDICTION
# =====================================================

if option == "Breast Cancer Prediction":

    st.markdown(
        '<div class="section-title">Breast Cancer Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Enter the required medical measurements to generate a '
        'machine learning prediction.'
        '</div>',
        unsafe_allow_html=True
    )

    model = load_breast_model()

    # Patient information

    st.markdown("#### PATIENT INFORMATION")

    p1, p2, p3 = st.columns(3)

    with p1:
        patient_name = st.text_input(
            "Patient Name",
            placeholder="Enter patient name"
        )

    with p2:
        patient_id = st.text_input(
            "Patient ID",
            placeholder="Enter patient ID"
        )

    with p3:
        patient_age = st.number_input(
            "Patient Age",
            min_value=1,
            max_value=120,
            value=30
        )

    st.markdown("---")

    st.markdown("#### MEDICAL MEASUREMENTS")

    col1, col2 = st.columns(2)

    with col1:

        radius_mean = st.number_input(
            "Radius Mean",
            min_value=0.0,
            value=12.0,
            step=0.1
        )

        perimeter_mean = st.number_input(
            "Perimeter Mean",
            min_value=0.0,
            value=80.0,
            step=0.1
        )

        smoothness_mean = st.number_input(
            "Smoothness Mean",
            min_value=0.0,
            value=0.09,
            step=0.01,
            format="%.4f"
        )

    with col2:

        texture_mean = st.number_input(
            "Texture Mean",
            min_value=0.0,
            value=15.0,
            step=0.1
        )

        area_mean = st.number_input(
            "Area Mean",
            min_value=0.0,
            value=500.0,
            step=1.0
        )

    st.markdown("")

    if st.button(
        "Generate Prediction",
        use_container_width=True
    ):

        data = np.array([[
            radius_mean,
            texture_mean,
            perimeter_mean,
            area_mean,
            smoothness_mean
        ]])

        prediction = model.predict(data)
        probability = model.predict_proba(data)

        confidence = max(probability[0]) * 100

        result = (
            "Benign (Non-Cancerous)"
            if prediction[0] == 0
            else "Malignant (Cancerous)"
        )

        st.markdown("---")

        st.markdown("### PREDICTION RESULTS")

        if prediction[0] == 0:

            st.markdown(f"""
            <div class="result-panel benign">
                <h3>Prediction Result: Benign</h3>
                <p>The model classified the supplied measurements
                as a benign result.</p>
                <h2 style="color:#16A34A;">
                    {confidence:.2f}%
                </h2>
                <p>Model confidence</p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-panel malignant">
                <h3>Prediction Result: Malignant</h3>
                <p>The model classified the supplied measurements
                as a malignant result.</p>
                <h2 style="color:#DC2626;">
                    {confidence:.2f}%
                </h2>
                <p>Model confidence</p>
            </div>
            """, unsafe_allow_html=True)

        # Report generation

        report = f"""
HEALTHCARE AI PLATFORM
BREAST CANCER PREDICTION REPORT

--------------------------------

PATIENT INFORMATION

Patient Name : {patient_name}
Patient ID   : {patient_id}
Patient Age  : {patient_age}

MEDICAL MEASUREMENTS

Radius Mean      : {radius_mean}
Texture Mean     : {texture_mean}
Perimeter Mean   : {perimeter_mean}
Area Mean        : {area_mean}
Smoothness Mean  : {smoothness_mean}

PREDICTION RESULTS

Prediction : {result}
Confidence : {confidence:.2f}%

--------------------------------

Generated by Healthcare AI Platform
Developer: Nikhil Nathan

IMPORTANT:
This machine learning output is for educational
and research purposes only. It is not a medical
diagnosis. Consult a qualified healthcare professional.
"""

        st.markdown("#### DOWNLOAD REPORT")

        st.download_button(
            label="Download Patient Report",
            data=report,
            file_name="Healthcare_AI_Patient_Report.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.warning(
            "This is an educational machine learning prediction, "
            "not a clinical diagnosis. Consult a qualified medical "
            "professional for medical decisions."
        )


# =====================================================
# INSURANCE SPEND PREDICTION
# =====================================================

elif option == "Insurance Spend Prediction":

    st.markdown(
        '<div class="section-title">Insurance Spend Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Estimate insurance expenditure using regional and '
        'economic indicators with machine learning.'
        '</div>',
        unsafe_allow_html=True
    )

    model = load_insurance_model()

    st.markdown("#### ECONOMIC AND REGIONAL INFORMATION")

    col1, col2 = st.columns(2)

    with col1:

        district = st.number_input(
            "District",
            min_value=1,
            value=1,
            step=1
        )

        region = st.selectbox(
            "Select Region",
            ["North", "South", "West"]
        )

    with col2:

        percapgdp = st.number_input(
            "Per Capita GDP",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

    st.markdown("")

    if st.button(
        "Generate Spend Prediction",
        use_container_width=True
    ):

        region_north = 1 if region == "North" else 0
        region_south = 1 if region == "South" else 0
        region_west = 1 if region == "West" else 0

        data = np.array([[
            district,
            percapgdp,
            region_north,
            region_south,
            region_west
        ]])

        prediction = model.predict(data)

        st.markdown("---")

        st.markdown("### PREDICTION RESULTS")

        st.markdown(f"""
        <div class="result-panel">
            <p>ESTIMATED INSURANCE SPEND</p>
            <h2 style="color:#2563EB; font-size:34px;">
                ₹ {prediction[0]:,.2f}
            </h2>
            <p>Machine learning predicted expenditure</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### INPUT SUMMARY")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("District", str(district))

        with r2:
            st.metric("Region", region)

        with r3:
            st.metric("Per Capita GDP", f"₹{percapgdp:,.0f}")


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer">

    <b>Healthcare AI Platform</b>

    <br><br>

    Developed by Nikhil Nathan

    <br>

    Built with Streamlit, Python, NumPy and Scikit-Learn

    <br><br>

    Machine Learning Based Predictive Analytics

</div>
""", unsafe_allow_html=True)
