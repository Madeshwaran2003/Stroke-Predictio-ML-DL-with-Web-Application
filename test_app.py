import requests
import json

url = 'http://127.0.0.1:5000/predict'
data = {
    "gender": "Male",
    "age": 67.0,
    "hypertension": 1,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
}

print("Testing prediction endpoint...")
response = requests.post(url, json=data)
print(f"Status Code: {response.status_code}")
try:
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(response.text)
