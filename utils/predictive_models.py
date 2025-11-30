# utils/predictive_models.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime, timedelta

class PolicyAnalytics:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        
    def analyze_resource_gaps(self, patient_df, facility_df):
        """Analyze gaps in healthcare resources"""
        
        # District-wise analysis
        district_analysis = patient_df.groupby('district').agg({
            'patient_id': 'count',
            'triage_color': lambda x: (x == 'RED').sum(),
            'transport_time_minutes': 'mean'
        }).rename(columns={
            'patient_id': 'total_cases',
            'triage_color': 'critical_cases',
            'transport_time_minutes': 'avg_transport_time'
        })
        
        # Facility capacity analysis
        facility_capacity = facility_df.groupby('district').agg({
            'icu_beds': 'sum',
            'general_beds': 'sum',
            'icu_available': 'sum'
        })
        
        # Merge analyses
        gap_analysis = district_analysis.merge(facility_capacity, 
                                             left_index=True, 
                                             right_index=True, 
                                             how='left').fillna(0)
        
        # Calculate gaps
        gap_analysis['icu_utilization'] = 1 - (gap_analysis['icu_available'] / gap_analysis['icu_beds']).replace([np.inf, -np.inf], 0)
        gap_analysis['critical_cases_per_bed'] = gap_analysis['critical_cases'] / gap_analysis['icu_beds'].replace(0, 1)
        
        return gap_analysis
    
    def predict_seasonal_demand(self, patient_df, months=6):
        """Predict seasonal variations in emergency demand"""
        
        # Add month and season
        patient_df['month'] = pd.to_datetime(patient_df['first_contact_time']).dt.month
        patient_df['season'] = patient_df['month'].apply(lambda x: 
            'Winter' if x in [12, 1, 2] else
            'Summer' if x in [3, 4, 5] else
            'Monsoon' if x in [6, 7, 8] else 'Autumn'
        )
        
        seasonal_trends = patient_df.groupby(['season', 'complaint']).size().unstack(fill_value=0)
        
        # Simple seasonal projection
        projections = {}
        current_month = datetime.now().month
        current_season = 'Winter' if current_month in [12, 1, 2] else \
                        'Summer' if current_month in [3, 4, 5] else \
                        'Monsoon' if current_month in [6, 7, 8] else 'Autumn'
        
        for complaint in seasonal_trends.columns:
            base_rate = seasonal_trends[complaint].mean()
            seasonal_factors = seasonal_trends[complaint] / base_rate
            projections[complaint] = {
                'current': base_rate * seasonal_factors.get(current_season, 1),
                'projected': {season: base_rate * factor for season, factor in seasonal_factors.items()}
            }
        
        return projections
    
    def identify_high_risk_corridors(self, patient_df, facility_df):
        """Identify high-risk transport corridors"""
        
        # Calculate transport efficiency
        transport_analysis = patient_df.groupby('district').agg({
            'transport_time_minutes': ['mean', 'max', 'count'],
            'triage_color': lambda x: (x == 'RED').sum()
        })
        
        transport_analysis.columns = ['avg_transport_time', 'max_transport_time', 'total_cases', 'critical_cases']
        
        # Identify corridors with long transport times for critical cases
        high_risk = transport_analysis[
            (transport_analysis['avg_transport_time'] > 60) | 
            (transport_analysis['critical_cases'] > transport_analysis['critical_cases'].quantile(0.75))
        ]
        
        return high_risk
    
    def generate_policy_recommendations(self, patient_df, facility_df):
        """Generate evidence-based policy recommendations"""
        
        recommendations = []
        
        # Resource gap analysis
        gap_analysis = self.analyze_resource_gaps(patient_df, facility_df)
        
        # ICU bed gaps
        high_icu_utilization = gap_analysis[gap_analysis['icu_utilization'] > 0.8]
        if not high_icu_utilization.empty:
            for district in high_icu_utilization.index:
                rec = {
                    'type': 'RESOURCE_ALLOCATION',
                    'priority': 'HIGH',
                    'district': district,
                    'recommendation': f'Increase ICU bed capacity in {district}. Current utilization: {high_icu_utilization.loc[district, "icu_utilization"]:.1%}',
                    'impact': 'Reduce critical case transfer delays'
                }
                recommendations.append(rec)
        
        # Transport efficiency
        transport_issues = self.identify_high_risk_corridors(patient_df, facility_df)
        for district in transport_issues.index:
            avg_time = transport_issues.loc[district, 'avg_transport_time']
            if avg_time > 90:
                rec = {
                    'type': 'INFRASTRUCTURE',
                    'priority': 'HIGH', 
                    'district': district,
                    'recommendation': f'Improve emergency transport infrastructure in {district}. Avg transport time: {avg_time:.0f}min',
                    'impact': 'Reduce mortality in time-sensitive emergencies'
                }
                recommendations.append(rec)
        
        # Seasonal preparedness
        seasonal_demand = self.predict_seasonal_demand(patient_df)
        for complaint, data in seasonal_demand.items():
            max_season = max(data['projected'], key=data['projected'].get)
            if data['projected'][max_season] > data['current'] * 1.3:
                rec = {
                    'type': 'SEASONAL_PLANNING',
                    'priority': 'MEDIUM',
                    'district': 'All',
                    'recommendation': f'Prepare for {max_season} surge in {complaint} cases. Expected increase: {(data["projected"][max_season]/data["current"]-1):.0%}',
                    'impact': 'Better resource allocation during peak seasons'
                }
                recommendations.append(rec)
        
        return recommendations

class EmergencyPredictor:
    def __init__(self):
        self.triage_model = None
        self.los_model = None
        self.policy_analytics = PolicyAnalytics()
        self.label_encoders = {}
        
    def prepare_triage_data(self, df):
        """Prepare data for triage prediction with Meghalaya context"""
        features = ['age', 'hr', 'sbp', 'rr', 'spo2', 'temp', 'complaint', 'district']
        X = df[features].copy()
        y = df['triage_color']
        
        # Encode categorical variables
        for col in ['complaint', 'district']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])
                
        return X, y
    
    def train_triage_model(self, df):
        """Train model to predict triage category"""
        X, y = self.prepare_triage_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.triage_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.triage_model.fit(X_train, y_train)
        
        accuracy = self.triage_model.score(X_test, y_test)
        
        # Feature importance for policy insights
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.triage_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return accuracy, feature_importance
    
    def predict_triage(self, patient_data):
        """Predict triage category for new patient"""
        if self.triage_model is None:
            return "Model not trained", None
            
        # Convert to dataframe
        patient_df = pd.DataFrame([patient_data])
        
        # Encode categorical variables
        for col in ['complaint', 'district']:
            if col in self.label_encoders and col in patient_df.columns:
                patient_df[col] = self.label_encoders[col].transform(patient_df[col])
        
        prediction = self.triage_model.predict(patient_df)[0]
        probabilities = self.triage_model.predict_proba(patient_df)[0]
        
        return prediction, probabilities
