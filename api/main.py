from utils import read_config,load_bins
from predict import *
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

config_dict = read_config("app_config.yaml")
model_path = config_dict["config"]["model_path"]
scalar_path = config_dict["config"]["scalar_path"]
cat_path = config_dict["config"]["cat_dict_path"]

try:
    print('Loading model....')
    scalar,cat_dict = load_bins(scalar_path,cat_path)
    model = mlflow.sklearn.load_model(model_path)
except Exception as e:
    print('Unable to load models')
    raise e

app = FastAPI()

class MapDtype(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
async def predict_data(data:MapDtype):
    data = data.dict()

    df = pd.DataFrame.from_dict(data,orient='index').T

    out,pred_label,pred_prob = predict(df,model,scalar,cat_dict)    
    prob = round(pred_prob[1],2)
    
    print(out,pred_label,prob)
    return {"predicted_class":str(out),
            "predicted_label":str(pred_label),
            "predicted_churn_probability":str(prob)}

if __name__=="__main__":
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)