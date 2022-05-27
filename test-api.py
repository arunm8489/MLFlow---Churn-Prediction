import json,requests

# local url
#url = 'https://lendersclub.herokuapp.com/predict'

url = 'http://127.0.0.1:4000/predict'


data = {"customerID":"1024-GUALD","gender":"Female","SeniorCitizen":"non senior","Partner":"Yes","Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":24.8,"TotalCharges":"24.8","Churn":"Yes"}

headers = {"Content-Type": "application/json; charset=utf-8"}

response = requests.post(url, headers = headers, json=data)
 

print(response)
print(response.text)