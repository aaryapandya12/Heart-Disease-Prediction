import streamlit as st
import pandas as pd
import joblib
from PIL import Image

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.set_page_config(
    page_title="Heart Health Guardian",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #FFF5F5;
    }
    .stSlider > div > div > div > div {
        background: #FF6B6B;
    }
    .st-bb {
        background-color: white;
    }
    .st-at {
        background-color: #FFE3E3;
    }
    .stButton>button {
        background-color: #FF6B6B;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        border-radius: 20px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF5252;
        transform: scale(1.05);
    }
    .header {
        color: #FF6B6B;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        font-size: 24px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .low-risk {
        background-color: #E3F9E5;
        color: #28A745;
    }
    .high-risk {
        background-color: #FFE3E3;
        color: #DC3545;
    }
    </style>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063172.png", width=150)
with col2:
    st.title("Heart Health Guardian")
    st.markdown("""
    <h3 class="header">Predict your risk of heart disease with our AI-powered tool</h3>
    """, unsafe_allow_html=True)

with st.expander("ℹ️ About this tool"):
    st.write("""
    This predictive tool uses machine learning to assess your risk of heart disease based on 
    key health indicators. The model was trained on clinical data from thousands of patients.
    
    **Note:** This tool is for informational purposes only and should not replace professional 
    medical advice. Always consult with your healthcare provider.
    """)

st.markdown("### 📋 Your Health Information")
form = st.form("prediction_form")

col1, col2 = form.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40, help="Select your current age")
    sex = st.selectbox("Sex", ["Male", "Female"], format_func=lambda x: x)
    chest_pain = st.selectbox("Chest Pain Type", 
                             ["Asymptomatic", "Atypical Angina", "Non-Anginal Pain", "Typical Angina"],
                             help="Type of chest pain experienced")
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120,
                                help="Your resting blood pressure measurement")
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200,
                                 help="Your cholesterol level")
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", 
                             ["No", "Yes"], 
                             help="Is your fasting blood sugar above 120?")

with col2:
    resting_ecg = st.selectbox("Resting ECG Results", 
                              ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                              help="Results of your resting electrocardiogram")
    max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150,
                      help="Highest heart rate achieved during exercise")
    exercise_angina = st.selectbox("Exercise-Induced Angina", 
                                 ["No", "Yes"],
                                 help="Do you experience angina during exercise?")
    oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, 0.1,
                       help="ST segment depression measured during exercise")
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                           ["Upsloping", "Flat", "Downsloping"],
                           help="The slope of the ST segment during peak exercise")

submit_button = form.form_submit_button("🔍 Assess My Heart Health")

sex_code = "M" if sex == "Male" else "F"
chest_pain_map = {
    "Asymptomatic": "ASY",
    "Atypical Angina": "ATA",
    "Non-Anginal Pain": "NAP",
    "Typical Angina": "TA"
}
chest_pain_code = chest_pain_map[chest_pain]
fasting_bs_code = 1 if fasting_bs == "Yes" else 0
resting_ecg_map = {
    "Normal": "Normal",
    "ST-T Wave Abnormality": "ST",
    "Left Ventricular Hypertrophy": "LVH"
}
resting_ecg_code = resting_ecg_map[resting_ecg]
exercise_angina_code = "Y" if exercise_angina == "Yes" else "N"
st_slope_map = {
    "Upsloping": "Up",
    "Flat": "Flat",
    "Downsloping": "Down"
}
st_slope_code = st_slope_map[st_slope]

if submit_button:
    with st.spinner("Analyzing your heart health..."):
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs_code,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex_code: 1,
            'ChestPainType_' + chest_pain_code: 1,
            'RestingECG_' + resting_ecg_code: 1,
            'ExerciseAngina_' + exercise_angina_code: 1,
            'ST_Slope_' + st_slope_code: 1
        }

        input_df = pd.DataFrame([raw_input])

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1] * 100

        st.balloons()
        st.markdown("## 🎯 Prediction Results")
        
        if prediction == 1:
            st.markdown(f"""
            <div class="result-box high-risk">
                ⚠️ High Risk of Heart Disease<br>
                <small>Probability: {probability:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.warning("""
            **Recommendations:**
            - Consult a cardiologist as soon as possible
            - Adopt a heart-healthy diet (Mediterranean diet recommended)
            - Begin a physician-approved exercise program
            - Monitor your blood pressure regularly
            - Consider stress-reduction techniques
            """)
        else:
            st.markdown(f"""
            <div class="result-box low-risk">
                ✅ Low Risk of Heart Disease<br>
                <small>Probability: {probability:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.success("""
            **Keep up the good work!**
            - Continue regular health check-ups
            - Maintain a balanced diet and active lifestyle
            - Monitor your heart health indicators annually
            - Avoid smoking and excessive alcohol consumption
            """)

        st.markdown("### 📊 Key Risk Factors")
        risk_factors = {
            'Age': age,
            'Blood Pressure': resting_bp,
            'Cholesterol': cholesterol,
            'Max Heart Rate': max_hr,
            'ST Depression': oldpeak
        }
        st.bar_chart(pd.DataFrame.from_dict(risk_factors, orient='index', columns=['Value']))

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>This tool is for educational purposes only. Created with ❤️ by Aarya</p>
    <p>Always consult with a healthcare professional for medical advice.</p>
</div>
""", unsafe_allow_html=True)