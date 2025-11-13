import streamlit as st
import pandas as pd
import pickle
import dill
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Travel Insurance Prediction",
    page_icon="✈️",
    layout="wide"
)

# Load model and explainer
@st.cache_resource
def load_model():
    with open('pipe_tuned_xgboost20251113_1122.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_explainer():
    with open('lime_explainer.dill', 'rb') as f:
        explainer = dill.load(f)
    return explainer

# Initialize
try:
    model = load_model()
    explainer = load_explainer()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model or explainer: {e}")
    model_loaded = False

# Title
st.title("✈️ Travel Insurance Claim Prediction")
st.markdown("---")

# Create input form in the center
col_left, col_center, col_right = st.columns([1, 3, 1])

with col_center:
    st.markdown("### Enter Trip and Customer Details")
    
    # Group 1: Trip Information
    st.markdown("#### 🌍 Trip Information")
    col1, col2 = st.columns(2)
    
    with col1:
        destination = st.selectbox(
            "Destination",
            options=['SINGAPORE', 'MALAYSIA', 'INDIA', 'UNITED STATES', 'KOREA, REPUBLIC OF', 
                    'THAILAND', 'GERMANY', 'JAPAN', 'INDONESIA', 'VIET NAM', 'AUSTRALIA', 
                    'FINLAND', 'UNITED KINGDOM', 'SRI LANKA', 'SPAIN', 'HONG KONG', 'MACAO', 
                    'CHINA', 'UNITED ARAB EMIRATES', 'IRAN, ISLAMIC REPUBLIC OF', 
                    'TAIWAN, PROVINCE OF CHINA', 'POLAND', 'CANADA', 'OMAN', 'PHILIPPINES', 
                    'GREECE', 'BELGIUM', 'TURKEY', 'BRUNEI DARUSSALAM', 'DENMARK', 
                    'SWITZERLAND', 'NETHERLANDS', 'SWEDEN', 'MYANMAR', 'KENYA', 
                    'CZECH REPUBLIC', 'FRANCE', 'RUSSIAN FEDERATION', 'PAKISTAN', 'ARGENTINA', 
                    'TANZANIA, UNITED REPUBLIC OF', 'SERBIA', 'ITALY', 'CROATIA', 
                    'NEW ZEALAND', 'PERU', 'MONGOLIA', 'CAMBODIA', 'QATAR', 'NORWAY', 
                    'LUXEMBOURG', 'MALTA', "LAO PEOPLE'S DEMOCRATIC REPUBLIC", 'ISRAEL', 
                    'SAUDI ARABIA', 'AUSTRIA', 'PORTUGAL', 'NEPAL', 'UKRAINE', 'ESTONIA', 
                    'ICELAND', 'BRAZIL', 'MEXICO', 'CAYMAN ISLANDS', 'PANAMA', 'BANGLADESH', 
                    'TURKMENISTAN', 'BAHRAIN', 'KAZAKHSTAN', 'TUNISIA', 'IRELAND', 'ETHIOPIA', 
                    'NORTHERN MARIANA ISLANDS', 'MALDIVES', 'SOUTH AFRICA', 'VENEZUELA', 
                    'COSTA RICA', 'JORDAN', 'MALI', 'CYPRUS', 'MAURITIUS', 'LEBANON', 
                    'KUWAIT', 'AZERBAIJAN', 'HUNGARY', 'BHUTAN', 'BELARUS', 'MOROCCO', 
                    'ECUADOR', 'UZBEKISTAN', 'CHILE', 'FIJI', 'PAPUA NEW GUINEA', 'ANGOLA', 
                    'FRENCH POLYNESIA', 'NIGERIA', 'MACEDONIA, THE FORMER YUGOSLAV REPUBLIC OF', 
                    'NAMIBIA', 'GEORGIA', 'COLOMBIA', 'SLOVENIA', 'EGYPT', 'ZIMBABWE', 
                    'BULGARIA', 'BERMUDA', 'URUGUAY', 'GUINEA', 'GHANA', 'BOLIVIA', 
                    'TRINIDAD AND TOBAGO', 'VANUATU', 'GUAM', 'UGANDA', 'JAMAICA', 'LATVIA', 
                    'ROMANIA', 'REPUBLIC OF MONTENEGRO', 'KYRGYZSTAN', 'GUADELOUPE', 'ZAMBIA', 
                    'RWANDA', 'BOTSWANA', 'GUYANA', 'LITHUANIA', 'GUINEA-BISSAU', 'SENEGAL', 
                    'CAMEROON', 'SAMOA', 'PUERTO RICO', 'TAJIKISTAN', 'ARMENIA', 
                    'FAROE ISLANDS', 'DOMINICAN REPUBLIC', 'MOLDOVA, REPUBLIC OF', 'BENIN', 
                    'REUNION'],
            index=0
        )
        
        duration = st.number_input(
            "Duration (days)",
            min_value=1,
            max_value=740,
            value=22,
            help="Trip duration in days"
        )
    
    with col2:
        product_name = st.selectbox(
            "Product Name",
            options=['Annual Silver Plan', 'Cancellation Plan', 'Basic Plan', 
                    '2 way Comprehensive Plan', 'Bronze Plan', '1 way Comprehensive Plan', 
                    'Rental Vehicle Excess Insurance', 'Single Trip Travel Protect Gold', 
                    'Silver Plan', 'Value Plan', '24 Protect', 'Annual Travel Protect Gold', 
                    'Comprehensive Plan', 'Ticket Protector', 'Travel Cruise Protect', 
                    'Single Trip Travel Protect Silver', 'Individual Comprehensive Plan', 
                    'Gold Plan', 'Annual Gold Plan', 'Child Comprehensive Plan', 
                    'Premier Plan', 'Annual Travel Protect Silver', 
                    'Single Trip Travel Protect Platinum', 'Annual Travel Protect Platinum', 
                    'Spouse or Parents Comprehensive Plan', 'Travel Cruise Protect Family'],
            index=0
        )
    
    st.markdown("---")
    
    # Group 2: Booking Details
    st.markdown("#### 🏢 Booking Details")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        agency = st.selectbox(
            "Agency",
            options=['C2B', 'EPX', 'JZI', 'CWT', 'LWC', 'ART', 'CSR', 'RAB', 'KML', 
                    'SSI', 'TST', 'TTW', 'ADM', 'CCR', 'CBH'],
            index=0
        )
    
    with col4:
        agency_type = st.selectbox(
            "Agency Type",
            options=['Airlines', 'Travel Agency'],
            index=0
        )
    
    with col5:
        distribution_channel = st.selectbox(
            "Distribution Channel",
            options=['Online', 'Offline'],
            index=0
        )
    
    st.markdown("---")
    
    # Group 3: Financial Information
    st.markdown("#### 💰 Financial Information")
    col6, col7 = st.columns(2)
    
    with col6:
        net_sales = st.number_input(
            "Net Sales",
            min_value=-357.50,
            max_value=682.00,
            value=26.00,
            step=1.0,
            help="Net sales amount"
        )
    
    with col7:
        commission = st.number_input(
            "Commission (in value)",
            min_value=0.0,
            max_value=262.76,
            value=0.0,
            step=0.1,
            help="Commission value"
        )
    
    st.markdown("---")
    
    # Group 4: Customer Information
    st.markdown("#### 👤 Customer Information")
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=88,
        value=36,
        help="Customer age"
    )
    
    st.markdown("---")
    
    # Predict button
    predict_button = st.button("🔮 Predict Claim Probability", use_container_width=True, type="primary")

# Prediction and LIME explanation
with col_center:
    if predict_button and model_loaded:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Agency': [agency],
            'Agency Type': [agency_type],
            'Distribution Channel': [distribution_channel],
            'Product Name': [product_name],
            'Destination': [destination],
            'Duration': [duration],
            'Net Sales': [net_sales],
            'Commision (in value)': [commission],
            'Age': [age]
        })
        
        try:
            # Make prediction
            prediction_proba = model.predict_proba(input_data)[0]
            prediction = model.predict(input_data)[0]
            
            # Display prediction
            st.markdown("### 📊 Prediction Results")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.metric(
                    label="Claim Prediction",
                    value="CLAIM" if prediction == 1 else "NO CLAIM",
                    delta=None
                )
            
            with result_col2:
                st.metric(
                    label="Claim Probability",
                    value=f"{prediction_proba[1]:.2%}",
                    delta=None
                )
            
            # Progress bar for probability
            st.progress(float(prediction_proba[1]))
            
            if prediction == 1:
                st.warning("⚠️ High probability of claim - Review policy carefully")
            else:
                st.success("✅ Low probability of claim - Standard processing")
            
            st.markdown("---")
            
            # LIME Explanation
            st.markdown("### 🔍 LIME Explanation")
            st.markdown("Understanding which features influenced this prediction:")
            
            # Transform input data using preprocessing step
            preprocessed_data = model.named_steps['preprocessing'].transform(input_data)
            
            # Generate LIME explanation
            explanation = explainer.explain_instance(
                preprocessed_data.values[0],
                model.named_steps['xgboost'].predict_proba,
                num_features=10
            )
            
            # Create and display the figure
            fig = explanation.as_pyplot_figure()
            fig.set_size_inches(10, 6)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Display feature importance as text
            with st.expander("📋 View Detailed Feature Contributions"):
                st.markdown("**Feature Impact on Prediction:**")
                for feature, weight in explanation.as_list():
                    direction = "increases" if weight > 0 else "decreases"
                    st.write(f"• {feature}: {direction} claim probability by {abs(weight):.4f}")
            
        except Exception as e:
            st.error(f"Error during prediction or explanation: {e}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Travel Insurance Claim Prediction System | Powered by XGBoost & LIME</p>
    </div>
    """,
    unsafe_allow_html=True
)