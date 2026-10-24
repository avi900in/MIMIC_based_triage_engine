import streamlit as st
import pandas as pd
import joblib
import os
import nlp_utils
import torch
import numpy as np
import pandas as pd
from train_multimodal_nn import LateFusionMultimodalNet
from transformers import AutoTokenizer

# Set page config
st.set_page_config(
    page_title="ED Triage Assistant",
    page_icon="🏥",
    layout="centered"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .reportview-container {
        background-color: #f4f6f9;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0052a3;
    }
    .high-risk {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        color: #cc0000;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        border: 2px solid #cc0000;
    }
    .low-risk {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        color: #006600;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        border: 2px solid #006600;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Emergency Department Triage Assistant")
st.write("""
This tool uses advanced machine learning trained on MIMIC-IV-ED data to predict whether an incoming patient is **High Risk** or **Standard Risk**.
""")

model_choice = st.selectbox("Select Model Engine", [
    "Deterministic Decision Tree (Calibrated)",
    "Transformer-Based Multimodal NN (PyTorch)",
    "TriageGeist (Comprehensive/Pediatric)",
    "SYNAPSE Recommendation (Transformer)",
    "Jev API (Cloud-Based Triage)"
])

jev_api_key = ""
if model_choice == "Jev API (Cloud-Based Triage)":
    jev_api_key = st.text_input("Enter Jev API Key", type="password")

@st.cache_resource
def load_tree_model():
    model_path = 'calibrated_triage_tree.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_pytorch_model():
    model = LateFusionMultimodalNet(num_vitals=7, vocab_size=30522, num_classes=5, dropout=0.0)
    model_path = 'best_multimodal_triage_net.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

tree_model = load_tree_model()
pytorch_model, pytorch_tokenizer = load_pytorch_model()

@st.cache_resource
def load_geist_model():
    model_path = 'triagegeist_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

geist_model = load_geist_model()

@st.cache_resource
def load_synapse_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_dir = './synapse_model'
    if os.path.exists(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        return model, tokenizer
    return None, None

synapse_model, synapse_tokenizer = load_synapse_model()

HX_COLS = [c for c in getattr(geist_model, 'feature_names_in_', []) if c.startswith('hx_')]

def compute_news2(rr, spo2, sbp, hr, temp_c, mental):
    """NEWS2 score (SpO2 scale 1, assumes room air)."""
    s = 0
    s += 3 if rr <= 8 else 1 if rr <= 11 else 0 if rr <= 20 else 2 if rr <= 24 else 3
    s += 3 if spo2 <= 91 else 2 if spo2 <= 93 else 1 if spo2 <= 95 else 0
    s += 3 if sbp <= 90 else 2 if sbp <= 100 else 1 if sbp <= 110 else 0 if sbp <= 219 else 3
    s += 3 if hr <= 40 else 1 if hr <= 50 else 0 if hr <= 90 else 1 if hr <= 110 else 2 if hr <= 130 else 3
    s += 3 if temp_c <= 35.0 else 1 if temp_c <= 36.0 else 0 if temp_c <= 38.0 else 1 if temp_c <= 39.0 else 2
    s += 0 if mental == 'alert' else 3
    return s

if tree_model is None:
    st.error("⚠️ Model file 'calibrated_triage_tree.pkl' not found. Please train the model first.")
else:
    st.header("Patient Presentation")
    
    # Form for input
    with st.form("patient_form"):
        chiefcomplaint = st.text_area("Chief Complaint (Symptoms & History)", placeholder="E.g. chest pain, shortness of breath, fall")
        
        st.subheader("Demographics & Conditions (SYNAPSE specific)")
        syn_col1, syn_col2, syn_col3 = st.columns(3)
        with syn_col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with syn_col2:
            duration = st.selectbox("Condition Duration", ["Less than 3 days", "Greater than 3 days", "1 Week", "More than 1 week"])
        with syn_col3:
            severity = st.selectbox("Severity", ["Mild", "Moderate", "Severe"])
            
        st.subheader("Vital Signs")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            temperature = st.number_input("Temperature (°F)", min_value=70.0, max_value=110.0, value=98.6)
            heartrate = st.number_input("Heart Rate (bpm)", min_value=30.0, max_value=250.0, value=80.0)
            pain = st.number_input("Pain Level (0-10)", min_value=0.0, max_value=10.0, value=0.0)
            
        with col2:
            resprate = st.number_input("Resp Rate (breaths/min)", min_value=5.0, max_value=60.0, value=16.0)
            o2sat = st.number_input("O2 Saturation (%)", min_value=50.0, max_value=100.0, value=98.0)
            
        with col3:
            sbp = st.number_input("Systolic BP", min_value=50.0, max_value=250.0, value=120.0)
            dbp = st.number_input("Diastolic BP", min_value=30.0, max_value=150.0, value=80.0)

        with st.expander("TriageGeist inputs (used by the TriageGeist engine)", expanded=(model_choice == "TriageGeist (Comprehensive/Pediatric)")):
            g1, g2, g3 = st.columns(3)
            with g1:
                g_sex = st.selectbox("Sex", ["M", "F", "Other"])
                g_arrival = st.selectbox("Arrival Mode", ["walk-in", "ambulance", "brought_by_family", "transfer", "police", "helicopter"])
                g_season = st.selectbox("Season", ["spring", "summer", "autumn", "winter"])
            with g2:
                g_mental = st.selectbox("Mental Status", ["alert", "confused", "agitated", "drowsy", "unresponsive"])
                g_gcs = st.number_input("GCS Total", min_value=3, max_value=15, value=15)
                g_system = st.selectbox("Complaint System", ["other", "cardiovascular", "respiratory", "neurological", "gastrointestinal", "trauma", "musculoskeletal", "infectious", "genitourinary", "psychiatric", "dermatological", "ENT", "ophthalmic", "endocrine"])
            with g3:
                g_bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=24.0)
                g_ed_visits = st.number_input("Prior ED visits (12m)", min_value=0, max_value=50, value=0)
                g_admits = st.number_input("Prior admissions (12m)", min_value=0, max_value=50, value=0)
                g_meds = st.number_input("Active medications", min_value=0, max_value=40, value=0)
            g_history = st.multiselect("Medical History", HX_COLS, format_func=lambda c: c[3:].replace('_', ' ').title())

        submitted = st.form_submit_button("Run Triage Assessment")
        
    if submitted:
        # Prepare input data
        input_data = pd.DataFrame({
            'temperature': [temperature],
            'heartrate': [heartrate],
            'resprate': [resprate],
            'o2sat': [o2sat],
            'sbp': [sbp],
            'dbp': [dbp],
            'pain': [pain],
            'chiefcomplaint': [chiefcomplaint]
        })
        
        with st.spinner("Analyzing..."):
            if model_choice == "Deterministic Decision Tree (Calibrated)":
                # Predict using Decision Tree
                pred = tree_model.predict(input_data)[0]
                prob = tree_model.predict_proba(input_data)[0]
                
                is_high_risk = (pred == 1)
                confidence = prob[1] if is_high_risk else prob[0]
                
            elif model_choice == "TriageGeist (Comprehensive/Pediatric)":
                if geist_model is None:
                    st.error("TriageGeist model not found. Train it first.")
                    st.stop()
                
                # Map inputs to what TriageGeist expects. We fill uncollected UI inputs with NaN/defaults so pipeline imputers handle them.
                temp_c = (temperature - 32) * 5.0/9.0
                news2 = compute_news2(resprate, o2sat, sbp, heartrate, temp_c, g_mental)
                geist_data = pd.DataFrame({
                    'temperature_c': [temp_c], # UI collects F
                    'heart_rate': [heartrate],
                    'respiratory_rate': [resprate],
                    'spo2': [o2sat],
                    'systolic_bp': [sbp],
                    'diastolic_bp': [dbp],
                    'mean_arterial_pressure': [dbp + (sbp - dbp)/3.0],
                    'pulse_pressure': [sbp - dbp],
                    'pain_score': [pain],
                    'shock_index': [heartrate / sbp if sbp > 0 else np.nan],
                    'gcs_total': [g_gcs],
                    'news2_score': [news2],
                    'bmi': [g_bmi],
                    'age': [age], 
                    'age_group': ['baby' if age < 3 else 'toddler' if age < 13 else 'teen' if age < 18 else 'adult' if age < 65 else 'senior'],
                    'num_prior_ed_visits_12m': [g_ed_visits],
                    'num_prior_admissions_12m': [g_admits],
                    'num_active_medications': [g_meds],
                    'num_comorbidities': [len(g_history)],
                    'sex': [g_sex],
                    'arrival_mode': [g_arrival],
                    'arrival_season': [g_season],
                    'mental_status_triage': [g_mental],
                    'chief_complaint_system': [g_system],
                    'chief_complaint_raw': [chiefcomplaint or "unknown"]
                })
                
                # History flags: 1 if selected, else 0
                for col in HX_COLS:
                    geist_data[col] = [1 if col in g_history else 0]
                
                # Ensure all training columns exist, in the trained order
                for col in geist_model.feature_names_in_:
                    if col not in geist_data.columns:
                        geist_data[col] = np.nan
                geist_data = geist_data[list(geist_model.feature_names_in_)]
                            
                pred = geist_model.predict(geist_data)[0]
                prob = geist_model.predict_proba(geist_data)[0]
                
                is_high_risk = (pred == 1)
                confidence = prob[1] if is_high_risk else prob[0]
                input_data = geist_data.drop(columns=HX_COLS).assign(medical_history=", ".join(g_history) or "none")
            
            elif model_choice == "SYNAPSE Recommendation (Transformer)":
                if synapse_model is None:
                    st.error("SYNAPSE model not found. Train it first.")
                    st.stop()
                
                # Format Text: Combine demographic + symptoms into a natural language sentence
                age_str = "below 5 years" if age < 5 else "6-15 years" if age <= 15 else "16-25 years" if age <= 25 else "26-40 years" if age <= 40 else "above 40 years"
                combined_text = f"Patient is a {age_str} {gender}. Condition duration: {duration}. Severity: {severity}. Symptoms: {chiefcomplaint}"
                
                import torch
                tokens = synapse_tokenizer(
                    combined_text,
                    padding='max_length',
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                
                with torch.no_grad():
                    outputs = synapse_model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
                    probs = torch.softmax(outputs.logits, dim=1)[0]
                    pred_class = torch.argmax(probs).item()
                
                # Classes: 0: Home Remedy, 1: Emergency Care
                is_high_risk = (pred_class == 1)
                confidence = probs[pred_class].item()
                
            elif model_choice == "Jev API (Cloud-Based Triage)":
                if not jev_api_key:
                    st.error("Please enter your Jev API key above.")
                    st.stop()
                    
                import requests
                headers = {
                    "Authorization": f"Bearer {jev_api_key}",
                    "Content-Type": "application/json"
                }
                
                patient_state = (
                    f"Age: {age}, Gender: {gender}, "
                    f"Chief Complaint: {chiefcomplaint}, "
                    f"Vitals: Temp {temperature}F, HR {heartrate}, RR {resprate}, SpO2 {o2sat}%, BP {sbp}/{dbp}, Pain {pain}/10"
                )
                
                payload = {
                    "state": patient_state,
                    "questions": {
                        "triage_risk": {
                            "type": "choice",
                            "instructions": "Evaluate the patient's triage risk based on presentation and vitals.",
                            "criteria": {
                                "High Risk": "Emergent, urgent, or high risk requiring immediate attention",
                                "Standard Risk": "Non-urgent, stable, or standard risk"
                            }
                        }
                    }
                }
                
                try:
                    response = requests.post("https://jev-ai.pro/api/v1/systemone", json=payload, headers=headers)
                    if response.status_code != 200:
                        st.error(f"Jev API returned status {response.status_code}: {response.text}")
                        st.stop()
                    result = response.json()
                    
                    answer = result.get("answers", {}).get("triage_risk", {})
                    chosen = answer.get("choice") or answer.get("value", "Standard Risk")
                    is_high_risk = (chosen == "High Risk")
                    confidence = answer.get("confidence", 0.0)
                except Exception as e:
                    st.error(f"Jev API error: {str(e)}")
                    st.stop()
                    
            else:
                # Predict using PyTorch Multimodal NN
                tokens = pytorch_tokenizer(
                    [chiefcomplaint],
                    padding='max_length',
                    truncation=True,
                    max_length=30,
                    return_tensors='pt'
                )
                text_tensor = tokens['input_ids']
                mask_tensor = tokens['attention_mask'].bool()
                
                # Standardize vitals using the fit scaler
                imputer = joblib.load('pytorch_vitals_imputer.pkl')
                scaler = joblib.load('pytorch_vitals_scaler.pkl')
                vitals_np = np.array([[temperature, heartrate, resprate, o2sat, sbp, dbp, pain]])
                vitals_np = imputer.transform(vitals_np)
                vitals_np = scaler.transform(vitals_np)
                
                vitals_tensor = torch.tensor(vitals_np, dtype=torch.float32)
                
                with torch.no_grad():
                    logits = pytorch_model(text_tensor, mask_tensor, vitals_tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    pred_class = torch.argmax(probs).item()
                
                # Assume classes 0,1 are high risk, 2,3,4 are standard for demo
                is_high_risk = pred_class in [0, 1]
                confidence = probs[pred_class].item()
            
            st.header("Triage Result")
            if is_high_risk:
                st.markdown(f'<div class="high-risk">⚠️ HIGH RISK<br>Confidence: {confidence*100:.1f}%</div>', unsafe_allow_html=True)
                st.info("Recommendation: Patient requires immediate or emergent care. Route to critical care bed.")
            else:
                st.markdown(f'<div class="low-risk">✅ STANDARD RISK<br>Confidence: {confidence*100:.1f}%</div>', unsafe_allow_html=True)
                st.info("Recommendation: Patient is stable for standard triage queue.")
                
        st.subheader("Model Decision Data")
        st.write("The model evaluated the following normalized feature vectors under its deterministic tree rules:")
        st.json(input_data.to_dict(orient='records')[0])
