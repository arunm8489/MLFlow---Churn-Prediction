import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib,pickle
from utils import read_config
import argparse,os


## feature engineering

def preprocess_data(df_train,df_test):
    df_train["Churn"] = df_train["Churn"].map({"Yes":1,"No":0})
    df_test["Churn"] = df_test["Churn"].map({"Yes":1,"No":0})

    df_train["SeniorCitizen"] = df_train["SeniorCitizen"].map({1:"senior",0:"non senior"})
    df_test["SeniorCitizen"] = df_test["SeniorCitizen"].map({1:"senior",0:"non senior"})


    df_train["InternetService_flag"] = df_train["InternetService"].apply(lambda x: "Yes" if x != "No" else "No")
    df_test["InternetService_flag"] = df_test["InternetService"].apply(lambda x: "Yes" if x != "No" else "No")

    # there are certain columns which are  ' ' in total charges column 
    df_train['TotalCharges'] = df_train['TotalCharges'].apply(lambda x: -1 if x == ' ' else float(x))
    df_train['TotalCharges'] = df_train['TotalCharges'].replace(-1,df_train['TotalCharges'].mean())
    df_test['TotalCharges'] = df_test['TotalCharges'].apply(lambda x: -1 if x == ' ' else float(x))
    df_test['TotalCharges'] = df_test['TotalCharges'].replace(-1,df_train['TotalCharges'].mean())

    return df_train,df_test


def process_num_cols(num_cols,df_train,
                df_test,scalar_path='artifacts/std_scaler.bin'):
    
    print('')
    print('Processing numerical columns')
    print('')

    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')

    std = StandardScaler()
    std.fit(df_train[num_cols])
    # saving std scalar
    joblib.dump(std, scalar_path, compress=True)

    df_train[num_cols] = std.transform(df_train[num_cols])
    df_test[num_cols] = std.transform(df_test[num_cols])

    return df_train,df_test
def generate_cat_dict(df_train,cat_cols,cat_save_path='artifacts/cat_values.bin'):
    cat_uniques = {}
    for col in cat_cols:
        ct = list(df_train[col].unique())
        cat_uniques[col] = ct

    with open(cat_save_path,'wb') as handle:
        joblib.dump(cat_uniques,handle,compress=True)

    return cat_uniques


def one_hot_encode(df_train,df_test,col_name,cat_dict):
    """
    onehot encode
    """
    known_cats = cat_dict[col_name]
    #train
    train_cat = pd.Categorical(df_train[col_name].values, categories = known_cats)
    train_cat = pd.get_dummies(train_cat,prefix=col_name)
    #test
    test_cat = pd.Categorical(df_test[col_name].values, categories = known_cats)
    test_cat = pd.get_dummies(test_cat,prefix=col_name)
  
    return train_cat, test_cat    


def process_categorical_data(cat_cols,df_train,df_test,cat_save_path):
    
    print(' ')
    print('  Processing categorical columns  .....')
    print(' ')
    # handling cat cols
    
    # generating dict with unique values in a column
    cat_dict = generate_cat_dict(df_train,cat_cols,cat_save_path)
 
    # one hot encoding
    train_results,test_results = [],[]
    for col in cat_cols:
        # print(col)
        train_cat,test_cat = one_hot_encode(df_train,df_test,col,cat_dict)
        train_results.append(train_cat),test_results.append(test_cat)

    df_train_cat = pd.concat(train_results,axis=1).reset_index(drop=True)
    df_test_cat = pd.concat(test_results,axis=1).reset_index(drop=True)

      
    print(f'Generated categoric features of shape - Train: {df_train_cat.shape}, Test: {df_test_cat.shape}')
    return df_train_cat,df_test_cat

def feature_pipeline(df_train,df_test,num_cols,
            cat_cols,scalar_save_path,cat_save_path):
    df_train,df_test = preprocess_data(df_train,df_test)
    df_train,df_test = process_num_cols(num_cols,df_train,
                df_test,scalar_save_path)
        
    df_train_cat,df_test_cat = process_categorical_data(cat_cols,df_train,
                        df_test,cat_save_path)

    dff_train = pd.concat([df_train[num_cols].reset_index(drop=True),
    df_train_cat.reset_index(drop=True),df_train["Churn"].reset_index(drop=True)],axis=1)
    dff_test = pd.concat([df_test[num_cols].reset_index(drop=True),
    df_test_cat.reset_index(drop=True),df_test["Churn"].reset_index(drop=True)],axis=1)
    print('Generated final datas of shape:')
    print(dff_train.shape,dff_test.shape)
    return dff_train,dff_test




if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()
    conf_dict = read_config(parsed_args.config)

    cat_save_path = conf_dict['config']['artifacts']['cat_col_dict_path']
    scalar_save_path = conf_dict['config']['artifacts']['std_scalar_path']
    data_dir_path = conf_dict['config']['artifacts']['data_dir_path']
    
    num_cols = ['tenure',"TotalCharges","MonthlyCharges"]
    cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
    'PaperlessBilling', 'PaymentMethod', 'InternetService_flag']

    try:
        print('Loading data from csv ....')
        df_train = pd.read_csv(os.path.join(data_dir_path,'train.csv'))
        df_test = pd.read_csv(os.path.join(data_dir_path,'test.csv'))
    except Exception as e:
        print('Unable to load train and test data.Please run load data pipe first')
        raise e

    try:
        dff_train,dff_test = feature_pipeline(df_train,df_test,num_cols,
            cat_cols,scalar_save_path,cat_save_path)

        dff_train.to_csv(os.path.join(data_dir_path,'train_processed.csv'),index=False)
        dff_test.to_csv(os.path.join(data_dir_path,'test_processed.csv'),index=False)
        print(' ')
        print(f'Saved generated data to {data_dir_path}')
        print(' ')
        print("**********************************************************")
        print('*********  preprocess data - run success   ***************')
        print("**********************************************************")
    except Exception as e:
        print('Unable to clean the data')
        raise e




    
