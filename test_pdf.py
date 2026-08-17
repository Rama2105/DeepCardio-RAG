from core.pdf_generator import generate_ecg_report, generate_arthritis_report
import sys
import traceback

def test_ecg_doctor():
    print("Testing ECG Report (Doctor)...")
    try:
        pdf_bytes = generate_ecg_report("Patient ECG seems normal. PR interval 150ms.", ["Guideline 1: Normal ECG"], 1.5, "doctor")
        with open("test_ecg_doctor.pdf", "wb") as f:
            f.write(pdf_bytes)
        print(" -> Success: test_ecg_doctor.pdf generated.")
    except Exception as e:
        print(" -> Error in ECG Doctor:")
        traceback.print_exc()
        raise

def test_ecg_patient():
    print("Testing ECG Report (Patient)...")
    try:
        pdf_bytes = generate_ecg_report("Everything is fine.", [], 1.0, "patient")
        with open("test_ecg_patient.pdf", "wb") as f:
            f.write(pdf_bytes)
        print(" -> Success: test_ecg_patient.pdf generated.")
    except Exception as e:
        print(" -> Error in ECG Patient:")
        traceback.print_exc()
        raise

def test_arthritis_doctor():
    print("Testing Arthritis Report (Doctor)...")
    try:
        patient_data = {"Age": 45, "Gender_M": 0, "Hb": 12.5, "ESRh": 15, "CRP": 4.5}
        prediction = {"risk_level": "LOW", "confidence": 0.85, "probabilities": {"low_risk": 0.85, "high_risk": 0.15}, "clinical_interpretation": "Patient is at low risk."}
        pdf_bytes = generate_arthritis_report(prediction, patient_data, None, None, "doctor")
        with open("test_arthritis_doctor.pdf", "wb") as f:
            f.write(pdf_bytes)
        print(" -> Success: test_arthritis_doctor.pdf generated.")
    except Exception as e:
        print(" -> Error in Arthritis Doctor:")
        traceback.print_exc()
        raise

def test_arthritis_patient():
    print("Testing Arthritis Report (Patient)...")
    try:
        patient_data = {"Age": 60, "Gender_M": 1}
        prediction = {"risk_level": "HIGH", "confidence": 0.92}
        pdf_bytes = generate_arthritis_report(prediction, patient_data, None, None, "patient")
        with open("test_arthritis_patient.pdf", "wb") as f:
            f.write(pdf_bytes)
        print(" -> Success: test_arthritis_patient.pdf generated.")
    except Exception as e:
        print(" -> Error in Arthritis Patient:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    test_ecg_doctor()
    test_ecg_patient()
    test_arthritis_doctor()
    test_arthritis_patient()
    print("All tests completed.")
