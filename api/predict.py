import mlflow 
import joblib
import pandas as pd

def feat_eng(df):
    df["InternetService_flag"] = df["InternetService"].apply(lambda x: "Yes" if x != "No" else "No")
    return df

def std_num_cols(df,num_cols,scalar):
    df[num_cols] = scalar.transform(df[num_cols])
    return df

def one_hot_cat_cols(df,cat_cols,cat_dict):
    results = []
    for col_name in cat_cols:
        known_cats = cat_dict[col_name]
        df_cat = pd.Categorical(df[col_name].values, categories = known_cats)
        df_cat = pd.get_dummies(df_cat,prefix=col_name)
        results.append(df_cat)
    df_cat_ = pd.concat(results,axis=1).reset_index(drop=True)
    return df_cat_

def preprocess_data(df,num_cols,cat_cols,scalar,cat_dict):
    df = feat_eng(df)
    df = std_num_cols(df,num_cols,scalar)
    df_cat = one_hot_cat_cols(df,cat_cols,cat_dict)
    dff = pd.concat([df[num_cols].reset_index(drop=True),
               df_cat.reset_index(drop=True)],axis=1)

    return dff

def predict(df,model,scalar_dict,cat_dict):
    num_cols = ['tenure',"TotalCharges","MonthlyCharges"]
    cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
    'PaperlessBilling', 'PaymentMethod', 'InternetService_flag']
    label_dict = {1:"Churn",0:"Not churn"}
    x = preprocess_data(df,num_cols,cat_cols,scalar_dict,cat_dict)
    
    out = model.predict(x)[0]
    pred_label = label_dict[out]
    pred_prob = model.predict_proba(x)[0]
    return out,pred_label,pred_prob


    