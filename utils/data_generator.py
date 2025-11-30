# utils/data_generator.py
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from .clinical_protocols import ICD_DIAGNOSES, get_icd_diagnoses

fake = Faker()

# Meghalaya-specific districts
MEGHALAYA_DISTRICTS = ['East Khasi Hills', 'West Garo Hills', 'East Garo Hills', 
                      'Ri-Bhoi', 'South West Khasi Hills', 'West Jaintia Hills', 
                      'East Jaintia Hills']

def generate_meghalaya_hospitals():
    """Generate Meghalaya-specific hospital data"""
    hospitals = [
        {
            'name': 'NEIGRIHMS, Shillong',
            'type': 'Public',
            'district': 'East Khasi Hills',
            'lat': 25.5788, 'lon': 91.8933,
            'icu_beds': 30, 'icu_available': random.randint(5, 25),
            'general_beds': 500, 'general_available': random.randint(100, 300),
            'specialties': ['Cardiology', 'Neurosurgery', 'Oncology', 'Multi-specialty'],
            'contact': '0364-2538010',
            'level': 'Tertiary'
        },
        {
            'name': 'Civil Hospital Shillong',
            'type': 'Public', 
            'district': 'East Khasi Hills',
            'lat': 25.5745, 'lon': 91.8793,
            'icu_beds': 8, 'icu_available': random.randint(1, 6),
            'general_beds': 300, 'general_available': random.randint(50, 200),
            'specialties': ['General Medicine', 'Surgery', 'Orthopedics'],
            'contact': '0364-2220105',
            'level': 'Secondary'
        },
        {
            'name': 'KJP Synod Hospital',
            'type': 'Private',
            'district': 'East Khasi Hills', 
            'lat': 25.5720, 'lon': 91.8815,
            'icu_beds': 6, 'icu_available': random.randint(1, 4),
            'general_beds': 150, 'general_available': random.randint(30, 100),
            'specialties': ['General Medicine', 'Surgery', 'Maternal'],
            'contact': '0364-2502111',
            'level': 'Secondary'
        },
        {
            'name': 'Tura Civil Hospital',
            'type': 'Public',
            'district': 'West Garo Hills',
            'lat': 25.5145, 'lon': 90.2201, 
            'icu_beds': 4, 'icu_available': random.randint(0, 3),
            'general_beds': 200, 'general_available': random.randint(40, 120),
            'specialties': ['General Medicine', 'Surgery', 'Emergency'],
            'contact': '03651-222247',
            'level': 'Secondary'
        },
        {
            'name': 'Mawphlang CHC',
            'type': 'Public',
            'district': 'East Khasi Hills',
            'lat': 25.4480, 'lon': 91.8510,
            'icu_beds': 0, 'icu_available': 0,
            'general_beds': 50, 'general_available': random.randint(10, 30),
            'specialties': ['Primary Care', 'Emergency'],
            'contact': '0364-2570001',
            'level': 'Primary'
        }
    ]
    return pd.DataFrame(hospitals)

def generate_synthetic_patients(n=1000):
    """Generate synthetic patient data with Meghalaya context"""
    patients = []
    
    complaints = ['Trauma', 'Maternal', 'Cardiac', 'Stroke', 'Respiratory', 'Sepsis']
    
    for i in range(n):
        # Basic patient info with Meghalaya context
        age = random.randint(18, 80)
        sex = random.choice(['M', 'F'])
        district = random.choice(MEGHALAYA_DISTRICTS)
        
        # Generate location within Meghalaya bounds
        if district == 'East Khasi Hills':
            lat = round(25.5 + random.random() * 0.2, 4)  # Shillong area
            lon = round(91.8 + random.random() * 0.2, 4)
        elif district == 'West Garo Hills':
            lat = round(25.4 + random.random() * 0.3, 4)  # Tura area
            lon = round(90.1 + random.random() * 0.3, 4)
        else:
            lat = round(25.3 + random.random() * 0.6, 4)
            lon = round(90.5 + random.random() * 1.0, 4)
        
        # Clinical parameters
        complaint = random.choice(complaints)
        
        # Get ICD diagnoses for this complaint
        icd_options = get_icd_diagnoses(complaint, 3)
        selected_diagnosis = random.choice(icd_options) if icd_options else {}
        
        # Generate vitals based on diagnosis severity
        if selected_diagnosis.get('severity') == 'HIGH':
            hr = random.randint(110, 160)
            sbp = random.randint(70, 100)
            rr = random.randint(24, 35)
            spo2 = random.randint(85, 92)
        else:
            hr = random.randint(60, 110)
            sbp = random.randint(90, 160)
            rr = random.randint(12, 22)
            spo2 = random.randint(92, 99)
            
        temp = round(random.uniform(36.0, 39.5), 1)
        
        # Calculate NEWS2 score
        news2 = calculate_news2(hr, sbp, rr, spo2, temp)
        
        # Determine triage color
        if news2 >= 7 or sbp < 90 or spo2 < 90:
            triage = 'RED'
        elif news2 >= 5:
            triage = 'YELLOW'
        else:
            triage = 'GREEN'
            
        # Timestamps
        onset_time = fake.date_time_between(start_date='-30d', end_date='now')
        contact_time = onset_time + timedelta(minutes=random.randint(5, 120))
        
        patient = {
            'patient_id': f'MEGH_{i:05d}',
            'age': age,
            'sex': sex,
            'district': district,
            'complaint': complaint,
            'icd_code': selected_diagnosis.get('code', 'R69'),
            'provisional_diagnosis': selected_diagnosis.get('diagnosis', 'Unknown'),
            'hr': hr,
            'sbp': sbp,
            'rr': rr,
            'spo2': spo2,
            'temp': temp,
            'news2_score': news2,
            'triage_color': triage,
            'onset_time': onset_time,
            'first_contact_time': contact_time,
            'location_lat': lat,
            'location_lon': lon,
            'outcome': random.choice(['Admitted', 'Discharged', 'Transferred', 'ICU', 'Expired']),
            'length_of_stay_hours': random.randint(1, 240),
            'transport_time_minutes': random.randint(15, 180)
        }
        patients.append(patient)
    
    return pd.DataFrame(patients)

def calculate_news2(hr, sbp, rr, spo2, temp):
    """Calculate NEWS2 score"""
    score = 0
    
    # Heart Rate
    if hr <= 40 or hr >= 131: score += 3
    elif hr >= 111 or hr <= 50: score += 1
    
    # Systolic BP
    if sbp <= 90: score += 3
    elif sbp <= 100: score += 2
    elif sbp >= 220: score += 3
    elif sbp >= 111: score += 0
    
    # Respiratory Rate
    if rr <= 8 or rr >= 25: score += 3
    elif rr >= 21: score += 1
    
    # SpO2
    if spo2 <= 91: score += 3
    elif spo2 <= 93: score += 2
    elif spo2 <= 95: score += 1
    
    # Temperature
    if temp <= 35.0: score += 3
    elif temp >= 39.1: score += 2
    elif temp >= 38.1: score += 1
    
    return score
