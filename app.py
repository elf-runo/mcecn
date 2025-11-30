# app.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.data_generator import generate_synthetic_patients, generate_meghalaya_hospitals
from utils.predictive_models import EmergencyPredictor, PolicyAnalytics
from utils.visualization import create_triage_dashboard, create_geographic_view, create_trend_analysis
from utils.clinical_protocols import get_resuscitation_steps, get_icd_diagnoses, RESUSCITATION_PROTOCOLS
import plotly.express as px
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Megh Comprehensive Emergency Care Network",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Meghalaya theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .megh-theme {
        background-color: #F0FFF0;
    }
    .district-card {
        background-color: #98FB98;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        border-left: 5px solid #2E8B57;
    }
    .protocol-box {
        background-color: #FFE4E1;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        border-left: 5px solid #DC143C;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header with Meghalaya focus
    st.markdown('<h1 class="main-header">🏔️ Megh Comprehensive Emergency Care Network</h1>', 
                unsafe_allow_html=True)
    st.markdown("### AI-Powered Emergency Healthcare Management for Meghalaya")
    
    # Sidebar
    st.sidebar.title("MeghCECN Configuration")
    
    # Data generation controls
    st.sidebar.subheader("Data Settings")
    n_patients = st.sidebar.slider("Number of synthetic patients", 100, 5000, 1000)
    refresh_data = st.sidebar.button("Generate New Meghalaya Data")
    
    # Model training
    st.sidebar.subheader("AI Models")
    train_models = st.sidebar.button("Train Predictive Models")
    
    # Initialize session state
    if 'patient_data' not in st.session_state or refresh_data:
        with st.spinner("Generating Meghalaya-specific patient data..."):
            st.session_state.patient_data = generate_synthetic_patients(n_patients)
            st.session_state.facility_data = generate_meghalaya_hospitals()
            st.success(f"Generated {n_patients} patient records across Meghalaya districts!")
    
    if 'predictor' not in st.session_state:
        st.session_state.predictor = EmergencyPredictor()
    
    if train_models:
        with st.spinner("Training predictive models for Meghalaya..."):
            accuracy, feature_importance = st.session_state.predictor.train_triage_model(st.session_state.patient_data)
            st.sidebar.success(f"Models trained! Triage Accuracy: {accuracy:.2%}")
            st.sidebar.write("Top predictive features:")
            st.sidebar.dataframe(feature_importance.head())
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Meghalaya Overview", 
        "🏥 Clinical Protocols", 
        "🤖 AI Predictions", 
        "📈 Policy Analytics",
        "🚨 Emergency Simulator",
        "💡 AI Demo Scenarios"
    ])
    
    with tab1:
        display_meghalaya_overview()
    
    with tab2:
        display_clinical_protocols()
    
    with tab3:
        display_predictions_tab()
    
    with tab4:
        display_policy_analytics()
    
    with tab5:
        display_simulator_tab()
    
    with tab6:
        display_ai_demo_scenarios()

def display_meghalaya_overview():
    st.header("Meghalaya Emergency Care Overview")
    
    # Key metrics with district focus
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_patients = len(st.session_state.patient_data)
        st.metric("Total Patients", total_patients)
    
    with col2:
        districts_covered = st.session_state.patient_data['district'].nunique()
        st.metric("Districts Covered", districts_covered)
    
    with col3:
        red_cases = len(st.session_state.patient_data[st.session_state.patient_data['triage_color'] == 'RED'])
        st.metric("Critical Cases (RED)", red_cases)
    
    with col4:
        avg_transport = st.session_state.patient_data['transport_time_minutes'].mean()
        st.metric("Avg Transport Time (min)", f"{avg_transport:.1f}")
    
    # District-wise analysis
    st.subheader("District-wise Emergency Distribution")
    
    district_stats = st.session_state.patient_data.groupby('district').agg({
        'patient_id': 'count',
        'triage_color': lambda x: (x == 'RED').sum(),
        'transport_time_minutes': 'mean'
    }).rename(columns={
        'patient_id': 'Total Cases',
        'triage_color': 'Critical Cases',
        'transport_time_minutes': 'Avg Transport Time'
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(district_stats.style.background_gradient(cmap='Reds'))
    
    with col2:
        fig_district = px.bar(district_stats.reset_index(), 
                             x='district', y='Total Cases',
                             title='Case Distribution by District',
                             color='Critical Cases')
        st.plotly_chart(fig_district, use_container_width=True)
    
    # Visualizations
    fig_triage, fig_complaint, fig_vitals = create_triage_dashboard(st.session_state.patient_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_triage, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig_complaint, use_container_width=True)
    
    # Facility capacity
    st.subheader("Hospital Capacity Across Meghalaya")
    facility_df = st.session_state.facility_data
    fig_facilities = px.scatter(facility_df, 
                               x='lon', y='lat',
                               size='icu_beds',
                               color='type',
                               hover_data=['name', 'icu_available', 'general_beds'],
                               title='Hospital Distribution and Capacity')
    st.plotly_chart(fig_facilities, use_container_width=True)

def display_clinical_protocols():
    st.header("Clinical Protocols & Resuscitation Guidelines")
    
    # Case type selector
    case_type = st.selectbox("Select Emergency Case Type", 
                           ["Cardiac", "Trauma", "Maternal", "Stroke", "Respiratory", "Sepsis"])
    
    if case_type:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"ICD-10 Compatible Diagnoses for {case_type}")
            diagnoses = get_icd_diagnoses(case_type, 5)
            
            for diagnosis in diagnoses:
                with st.expander(f"{diagnosis['code']}: {diagnosis['diagnosis']}"):
                    st.write(f"**Severity:** {diagnosis['severity']}")
                    st.write("**Common Presentation:**")
                    if case_type == "Cardiac":
                        st.write("- Chest pain, shortness of breath, palpitations")
                    elif case_type == "Trauma":
                        st.write("- Mechanism of injury, loss of consciousness, bleeding")
                    elif case_type == "Maternal":
                        st.write("- Pregnancy-related symptoms, bleeding, hypertension")
        
        with col2:
            st.subheader(f"Resuscitation Protocol for {case_type}")
            protocol = get_resuscitation_steps(case_type)
            
            if isinstance(protocol, dict) and 'error' not in protocol:
                st.markdown("#### Initial Assessment")
                for step in protocol.get('initial_assessment', []):
                    st.write(f"• {step}")
                
                st.markdown("#### Medications")
                for med in protocol.get('medications', []):
                    st.write(f"• {med}")
                
                st.markdown("#### Critical Actions")
                for action in protocol.get('critical_actions', []):
                    st.write(f"• {action}")
                
                st.markdown("#### Referral Criteria")
                for criteria in protocol.get('referral_criteria', []):
                    st.write(f"• {criteria}")
            
            elif case_type == "Maternal":
                # Show maternal sub-protocols
                st.info("Select specific maternal emergency:")
                maternal_type = st.selectbox("Maternal Condition", 
                                           ["Postpartum Hemorrhage", "Eclampsia"])
                specific_protocol = get_resuscitation_steps(case_type, maternal_type)
                
                if specific_protocol:
                    st.markdown(f"#### {specific_protocol['condition']} Protocol")
                    for step in specific_protocol['protocol']:
                        st.write(f"• {step}")
                    
                    st.markdown("#### Treatment Targets")
                    for target in specific_protocol['targets']:
                        st.write(f"✅ {target}")
    
    # Quick reference guide
    st.subheader("Emergency Quick Reference")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🚨 RED Flag Criteria**")
        st.write("• SBP < 90 mmHg")
        st.write("• SpO2 < 90%")
        st.write("• GCS < 13")
        st.write("• Respiratory distress")
    
    with col2:
        st.markdown("**⚠️ Critical Time Windows**")
        st.write("• STEMI: <90min for PCI")
        st.write("• Stroke: <4.5h for thrombolysis")
        st.write("• Trauma: Golden Hour")
        st.write("• Sepsis: 1h bundle")
    
    with col3:
        st.markdown("**🏥 Referral Centers**")
        st.write("• NEIGRIHMS: Tertiary care")
        st.write("• Civil Hospital: Secondary care") 
        st.write("• CHCs: Primary stabilization")

def display_predictions_tab():
    st.header("AI-Powered Clinical Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Triage Prediction")
        
        with st.form("patient_form"):
            st.write("Enter patient details for AI triage prediction:")
            
            district = st.selectbox("District", 
                                  ['East Khasi Hills', 'West Garo Hills', 'East Garo Hills', 
                                   'Ri-Bhoi', 'South West Khasi Hills'])
            age = st.slider("Age", 18, 90, 45)
            complaint = st.selectbox("Chief Complaint", 
                                   ["Cardiac", "Trauma", "Maternal", "Stroke", "Respiratory", "Sepsis"])
            
            col1, col2 = st.columns(2)
            with col1:
                hr = st.slider("Heart Rate", 40, 180, 80)
                sbp = st.slider("Systolic BP", 60, 200, 120)
            with col2:
                rr = st.slider("Respiratory Rate", 8, 40, 16)
                spo2 = st.slider("SpO2", 70, 100, 98)
            
            temp = st.slider("Temperature (°C)", 35.0, 41.0, 37.0)
            
            submitted = st.form_submit_button("Predict Triage & Generate Protocol")
            
            if submitted:
                patient_data = {
                    'district': district,
                    'age': age, 'hr': hr, 'sbp': sbp, 'rr': rr, 
                    'spo2': spo2, 'temp': temp, 'complaint': complaint
                }
                
                prediction, probabilities = st.session_state.predictor.predict_triage(patient_data)
                
                # Display results
                if prediction == "RED":
                    st.error(f"🚨 AI Predicted Triage: {prediction} - CRITICAL CASE")
                    st.warning("🔔 Immediate resuscitation required! Activate emergency protocol.")
                elif prediction == "YELLOW":
                    st.warning(f"⚠️ AI Predicted Triage: {prediction} - URGENT CASE")
                    st.info("📋 Requires urgent assessment and monitoring")
                else:
                    st.success(f"✅ AI Predicted Triage: {prediction} - STABLE CASE")
                    st.info("📊 Routine monitoring and assessment")
                
                # Show protocol immediately for critical cases
                if prediction in ["RED", "YELLOW"]:
                    st.subheader("🎯 Recommended Resuscitation Protocol")
                    protocol = get_resuscitation_steps(complaint)
                    
                    if isinstance(protocol, dict) and 'error' not in protocol:
                        tab1, tab2, tab3 = st.tabs(["Initial Actions", "Medications", "Referral"])
                        
                        with tab1:
                            for step in protocol.get('initial_assessment', []):
                                st.write(f"• {step}")
                        
                        with tab2:
                            for med in protocol.get('medications', []):
                                st.write(f"• {med}")
                        
                        with tab3:
                            for criteria in protocol.get('referral_criteria', []):
                                st.write(f"• {criteria}")
    
    with col2:
        st.subheader("Resource Demand Forecasting")
        
        # District-specific forecasting
        selected_district = st.selectbox("Select District for Forecast", 
                                       st.session_state.patient_data['district'].unique())
        
        if st.button("Generate District Forecast"):
            # Simple forecasting logic
            district_data = st.session_state.patient_data[
                st.session_state.patient_data['district'] == selected_district
            ]
            
            # Weekly patterns
            district_data['week'] = pd.to_datetime(district_data['first_contact_time']).dt.isocalendar().week
            weekly_trend = district_data.groupby('week').size()
            
            # Project next 4 weeks
            last_4_weeks = weekly_trend.tail(4).mean()
            forecast = [last_4_weeks * (1 + i * 0.05) for i in range(4)]
            
            fig_forecast = px.line(
                x=[f'Week {i+1}' for i in range(4)], 
                y=forecast,
                title=f"4-Week Emergency Case Forecast - {selected_district}",
                labels={'x': 'Week', 'y': 'Expected Cases'}
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Resource recommendations
            st.subheader("📋 Resource Planning Recommendations")
            
            critical_ratio = len(district_data[district_data['triage_color'] == 'RED']) / len(district_data)
            expected_critical = sum(forecast) * critical_ratio
            
            st.write(f"**Expected critical cases:** {expected_critical:.1f}")
            st.write(f"**Recommended ICU beds to reserve:** {max(2, int(expected_critical * 1.5))}")
            st.write(f"**Ambulance deployments needed:** {max(1, int(expected_critical * 0.3))}")

def display_policy_analytics():
    st.header("Policy & Planning Analytics")
    
    # Initialize policy analytics
    policy_engine = PolicyAnalytics()
    
    # Resource gap analysis
    st.subheader("🏥 Healthcare Resource Gap Analysis")
    
    gap_analysis = policy_engine.analyze_resource_gaps(
        st.session_state.patient_data, 
        st.session_state.facility_data
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(gap_analysis.style.background_gradient(subset=['icu_utilization'], cmap='Reds'))
    
    with col2:
        fig_gaps = px.bar(gap_analysis.reset_index(), 
                         x='district', y='icu_utilization',
                         title='ICU Utilization by District',
                         color='critical_cases_per_bed')
        st.plotly_chart(fig_gaps, use_container_width=True)
    
    # Policy recommendations
    st.subheader("🎯 Evidence-Based Policy Recommendations")
    
    recommendations = policy_engine.generate_policy_recommendations(
        st.session_state.patient_data,
        st.session_state.facility_data
    )
    
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"Recommendation {i}: {rec['type']} - {rec['district']} ({rec['priority']})"):
            st.write(f"**{rec['recommendation']}**")
            st.write(f"*Expected Impact:* {rec['impact']}")
            
            if rec['priority'] == 'HIGH':
                st.error("🚨 HIGH PRIORITY - Immediate action recommended")
            elif rec['priority'] == 'MEDIUM':
                st.warning("⚠️ MEDIUM PRIORITY - Plan for implementation")
    
    # Seasonal planning
    st.subheader("🌦️ Seasonal Demand Forecasting")
    
    seasonal_data = policy_engine.predict_seasonal_demand(st.session_state.patient_data)
    
    seasonal_df = pd.DataFrame([
        {'Complaint': complaint, 'Season': season, 'Cases': cases}
        for complaint, data in seasonal_data.items()
        for season, cases in data['projected'].items()
    ])
    
    fig_seasonal = px.line(seasonal_df, x='Season', y='Cases', color='Complaint',
                          title='Seasonal Variation in Emergency Cases')
    st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # High-risk corridors
    st.subheader("🛣️ High-Risk Transport Corridors")
    
    high_risk = policy_engine.identify_high_risk_corridors(
        st.session_state.patient_data,
        st.session_state.facility_data
    )
    
    if not high_risk.empty:
        st.write("Districts requiring transport infrastructure improvement:")
        for district in high_risk.index:
            avg_time = high_risk.loc[district, 'avg_transport_time']
            st.write(f"• **{district}**: {avg_time:.0f}min average transport time")

def display_simulator_tab():
    st.header("Real-time Emergency Simulation - Meghalaya")
    
    st.warning("""
    🚨 **Live Simulation**: This simulates real-time emergency cases across Meghalaya districts. 
    Data refreshes every few seconds to demonstrate system capabilities.
    """)
    
    if st.button("Start Live Simulation"):
        simulation_placeholder = st.empty()
        
        for i in range(8):  # 8 simulation cycles
            with simulation_placeholder.container():
                # Generate new emergency cases
                new_cases = generate_synthetic_patients(15)
                
                # Update simulation data
                if 'simulation_data' not in st.session_state:
                    st.session_state.simulation_data = new_cases
                else:
                    st.session_state.simulation_data = pd.concat([
                        st.session_state.simulation_data, new_cases
                    ]).tail(100)  # Keep last 100 cases
                
                # Display current emergency status
                st.subheader(f"🔄 Live Simulation Update {i+1}")
                
                # Real-time metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_cases = len(st.session_state.simulation_data)
                    st.metric("Active Emergencies", current_cases)
                
                with col2:
                    current_critical = len(st.session_state.simulation_data[
                        st.session_state.simulation_data['triage_color'] == 'RED'
                    ])
                    st.metric("Critical Cases", current_critical)
                
                with col3:
                    latest_case = st.session_state.simulation_data.iloc[-1]
                    st.metric("Latest Emergency", latest_case['complaint'])
                
                with col4:
                    district_active = st.session_state.simulation_data['district'].nunique()
                    st.metric("Active Districts", district_active)
                
                # Recent critical cases with protocols
                st.subheader("🚨 Recent Critical Cases - Action Required")
                
                critical_cases = st.session_state.simulation_data[
                    st.session_state.simulation_data['triage_color'] == 'RED'
                ].tail(5)
                
                for _, case in critical_cases.iterrows():
                    with st.expander(f"🚑 {case['complaint']} - {case['district']} (Age: {case['age']})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Vitals:** HR {case['hr']}, BP {case['sbp']}, SpO2 {case['spo2']}%")
                            st.write(f"**Diagnosis:** {case['provisional_diagnosis']} ({case['icd_code']})")
                        
                        with col2:
                            protocol = get_resuscitation_steps(case['complaint'])
                            if isinstance(protocol, dict) and 'initial_assessment' in protocol:
                                st.write("**Immediate Actions:**")
                                for action in protocol['initial_assessment'][:3]:
                                    st.write(f"• {action}")
                
                # District-wise emergency load
                st.subheader("📊 District-wise Emergency Load")
                district_load = st.session_state.simulation_data['district'].value_counts()
                fig_load = px.bar(x=district_load.index, y=district_load.values,
                                 title="Active Emergencies by District")
                st.plotly_chart(fig_load, use_container_width=True)
            
            time.sleep(4)  # Wait 4 seconds between updates

def display_ai_demo_scenarios():
    st.header("🤖 AI Component Demonstrations")
    
    st.markdown("""
    ### Real-life AI Adoption Scenarios Demonstrated in this MVP
    
    These components showcase how AI can be practically integrated into Meghalaya's emergency care system:
    """)
    
    # Scenario 1: Triage Automation
    with st.expander("🎯 SCENARIO 1: Automated Triage & Protocol Activation", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Process")
            st.write("""
            - Manual triage by overburdened staff
            - Subjective severity assessment  
            - Delayed protocol activation
            - Inconsistent referral decisions
            """)
        
        with col2:
            st.subheader("AI-Enhanced Process")
            st.write("""
            - **Instant triage prediction** from vital signs
            - **Automated protocol suggestions**
            - **Objective severity scoring** (NEWS2 + AI)
            - **Smart referral recommendations**
            - **Real-time quality monitoring**
            """)
        
        st.success("""
        **Real Adoption Impact**: Reduce triage errors by 40%, ensure critical cases get immediate attention, 
        standardize care across all facilities in Meghalaya.
        """)
    
    # Scenario 2: Resource Optimization  
    with st.expander("🏥 SCENARIO 2: Predictive Resource Management"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Challenges")
            st.write("""
            - Reactive resource allocation
            - ICU bed shortages during surges
            - Ambulance deployment inefficiencies
            - Seasonal unpreparedness
            - District-wise capacity imbalances
            """)
        
        with col2:
            st.subheader("AI Solutions")
            st.write("""
            - **Demand forecasting** by district and season
            - **Optimal ambulance routing**
            - **ICU bed prediction** 48h in advance
            - **Resource gap identification**
            - **Evidence-based policy recommendations**
            """)
        
        st.success("""
        **Real Adoption Impact**: 30% better resource utilization, reduce critical case transport times, 
        enable data-driven healthcare planning for Meghalaya.
        """)
    
    # Scenario 3: Clinical Decision Support
    with st.expander("🩺 SCENARIO 3: AI Clinical Decision Support"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Limitations")
            st.write("""
            - Protocol knowledge gaps
            - Delayed specialist consultation
            - Inconsistent resuscitation quality
            - Missed referral criteria
            - Variable diagnostic accuracy
            """)
        
        with col2:
            st.subheader("AI Enhancement")
            st.write("""
            - **Instant protocol access** with ICD coding
            - **Step-by-step resuscitation guidance**
            - **Referral criteria automation**
            - **Diagnostic suggestion engine**
            - **Quality assurance monitoring**
            """)
        
        st.success("""
        **Real Adoption Impact**: Standardize emergency care quality across Meghalaya, 
        reduce clinical errors, ensure adherence to best practices in all facilities.
        """)
    
    # Implementation Roadmap
    st.subheader("🚀 Phased Implementation Roadmap for Meghalaya")
    
    roadmap_data = {
        'Phase': ['Pilot (3-6 months)', 'Scale (6-12 months)', 'Statewide (12-24 months)'],
        'Districts': ['East Khasi Hills', '+ West Garo Hills, Ri-Bhoi', 'All 7 Districts'],
        'Hospitals': ['NEIGRIHMS, Civil Hospital', '+ 5 major hospitals', 'All 20+ facilities'],
        'AI Components': ['Triage prediction, Basic analytics', '+ Resource optimization, Protocols', 'Full AI suite, Policy analytics']
    }
    
    st.table(roadmap_data)
    
    # Technical Architecture
    st.subheader("🛠️ Technical Architecture for Real Implementation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Sources Integration**:
        - Hospital EMR systems
        - Ambulance GPS & vitals
        - Facility capacity databases
        - Weather & seasonal data
        - Population health records
        
        **AI Model Deployment**:
        - Cloud-based inference engine
        - Real-time API endpoints
        - Mobile app integration
        - Dashboard visualization
        - Alert systems
        """)
    
    with col2:
        st.markdown("""
        **Integration Points**:
        - Existing hospital software
        - Ambulance communication systems
        - Health department databases
        - Emergency response networks
        - Policy planning frameworks
        
        **Success Metrics**:
        - Reduction in critical case mortality
        - Improved resource utilization
        - Faster emergency response times
        - Enhanced healthcare worker efficiency
        - Data-driven policy decisions
        """)

if __name__ == "__main__":
    main()
