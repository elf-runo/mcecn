# utils/clinical_protocols.py
import pandas as pd

# ICD-10 compatible provisional diagnoses
ICD_DIAGNOSES = {
    'Cardiac': [
        {'code': 'I21.9', 'diagnosis': 'Acute myocardial infarction, unspecified', 'severity': 'HIGH'},
        {'code': 'I50.9', 'diagnosis': 'Heart failure, unspecified', 'severity': 'HIGH'},
        {'code': 'I48.91', 'diagnosis': 'Unspecified atrial fibrillation', 'severity': 'MEDIUM'},
        {'code': 'I10', 'diagnosis': 'Essential hypertension', 'severity': 'MEDIUM'}
    ],
    'Trauma': [
        {'code': 'S06.9', 'diagnosis': 'Intracranial injury, unspecified', 'severity': 'HIGH'},
        {'code': 'S22.9', 'diagnosis': 'Fracture of rib, unspecified', 'severity': 'MEDIUM'},
        {'code': 'T14.8', 'diagnosis': 'Other injuries of unspecified body region', 'severity': 'VARIABLE'}
    ],
    'Maternal': [
        {'code': 'O72.1', 'diagnosis': 'Other immediate postpartum hemorrhage', 'severity': 'HIGH'},
        {'code': 'O15.9', 'diagnosis': 'Eclampsia, unspecified', 'severity': 'HIGH'},
        {'code': 'O60.1', 'diagnosis': 'Preterm labor with preterm delivery', 'severity': 'HIGH'}
    ],
    'Stroke': [
        {'code': 'I63.9', 'diagnosis': 'Cerebral infarction, unspecified', 'severity': 'HIGH'},
        {'code': 'I61.9', 'diagnosis': 'Intracerebral hemorrhage, unspecified', 'severity': 'HIGH'}
    ],
    'Respiratory': [
        {'code': 'J18.9', 'diagnosis': 'Pneumonia, unspecified', 'severity': 'HIGH'},
        {'code': 'J44.9', 'diagnosis': 'Chronic obstructive pulmonary disease, unspecified', 'severity': 'MEDIUM'},
        {'code': 'J96.00', 'diagnosis': 'Acute respiratory failure, unspecified', 'severity': 'HIGH'}
    ],
    'Sepsis': [
        {'code': 'A41.9', 'diagnosis': 'Sepsis, unspecified organism', 'severity': 'HIGH'},
        {'code': 'R65.20', 'diagnosis': 'Severe sepsis without septic shock', 'severity': 'HIGH'}
    ]
}

# Resuscitation protocols for each case type
RESUSCITATION_PROTOCOLS = {
    'Cardiac': {
        'initial_assessment': ['ABC assessment', '12-lead ECG', 'IV access', 'Cardiac monitoring'],
        'medications': ['Aspirin 300mg', 'Clopidogrel 300mg', 'Morphine for pain', 'Oxygen if hypoxic'],
        'critical_actions': ['STEMI activation if indicated', 'Prepare for thrombolysis/PCI', 'Monitor for arrhythmias'],
        'referral_criteria': ['STEMI on ECG', 'Hemodynamic instability', 'Refractory chest pain']
    },
    'Trauma': {
        'initial_assessment': ['Primary survey (ABCDE)', 'C-spine immobilization', 'Hemorrhage control', 'FAST scan'],
        'medications': ['Tranexamic acid 1g IV', 'Analgesia as needed', 'Tetanus prophylaxis'],
        'critical_actions': ['Massive transfusion protocol if needed', 'Prepare for emergency surgery', 'Head CT if head injury'],
        'referral_criteria': ['GCS <13', 'SBP <90', 'Penetrating trauma to torso', 'Unstable pelvic fracture']
    },
    'Maternal': [
        {
            'condition': 'Postpartum Hemorrhage',
            'protocol': [
                'Call for help - activate PPH protocol',
                'Uterine massage',
                'Oxytocin 40 units in 1L NS at 125ml/hr',
                'Misoprostol 800mcg PR',
                'Tranexamic acid 1g IV',
                'Consider intrauterine balloon tamponade'
            ],
            'targets': ['Control bleeding within 15min', 'SBP >90', 'HR <120']
        },
        {
            'condition': 'Eclampsia',
            'protocol': [
                'Magnesium sulfate loading 4g IV over 20min',
                'Then 1g/hr maintenance',
                'BP control with labetalol/hydralazine',
                'Delivery planning'
            ],
            'targets': ['Seizure control', 'BP <160/110', 'Prepare for delivery']
        }
    ],
    'Stroke': {
        'initial_assessment': ['NIHSS assessment', 'Non-contrast CT head', 'Blood glucose check'],
        'medications': ['Consider thrombolysis if within window', 'Aspirin if hemorrhagic excluded'],
        'critical_actions': ['Neurology consult', 'Monitor for deterioration', 'Swallow assessment'],
        'referral_criteria': ['NIHSS >5', 'Within thrombolysis window', 'Hemorrhagic stroke needing neurosurgery']
    },
    'Sepsis': {
        'initial_assessment': ['Quick SOFA assessment', 'Lactate level', 'Blood cultures', 'Source identification'],
        'medications': ['Broad-spectrum antibiotics within 1hr', 'IV fluids bolus', 'Vasopressors if needed'],
        'critical_actions': ['Measure lactate', 'Administer antibiotics', 'Fluid resuscitation', 'Source control'],
        'referral_criteria': ['Lactate >4', 'Need for vasopressors', 'Organ dysfunction']
    }
}

def get_resuscitation_steps(case_type, specific_diagnosis=None):
    """Get appropriate resuscitation steps based on case type and diagnosis"""
    if case_type in RESUSCITATION_PROTOCOLS:
        if case_type == 'Maternal' and specific_diagnosis:
            # Return specific maternal protocol
            for protocol in RESUSCITATION_PROTOCOLS[case_type]:
                if protocol['condition'].lower() in specific_diagnosis.lower():
                    return protocol
        return RESUSCITATION_PROTOCOLS[case_type]
    return {"error": "No protocol found for this case type"}

def get_icd_diagnoses(case_type, count=3):
    """Get ICD-compatible diagnoses for a case type"""
    if case_type in ICD_DIAGNOSES:
        return ICD_DIAGNOSES[case_type][:count]
    return []
