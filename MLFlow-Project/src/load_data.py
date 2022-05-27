import numpy as np
import pandas as pd
import os
from utils import read_config
from sklearn.model_selection import train_test_split
import argparse

def load_and_split(data_path,random_state):
    try:
        data = pd.read_csv(data_path)
        X = data.drop(["Churn"], axis=1)
        y = data['Churn']
        X_train,X_test,y_train, y_test =  train_test_split(X,y,test_size=0.25,random_state=random_state)
        df_train = pd.concat([X_train,y_train],axis=1)
        df_test = pd.concat([X_test,y_test],axis=1) 
        return df_train,df_test

    except Exception as e:
        print(f'Unable to load data.:{e}')
        raise e





if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    conf_dict = read_config(parsed_args.config)

    data_path = conf_dict['config']['data_path']
    random_state = conf_dict['config']['random_state']
    target_data_path = conf_dict['config']['artifacts']['data_dir_path']

    # make artifact directory if not exisits
    if not os.path.exists(target_data_path):
        os.makedirs(target_data_path)
    
    try: 
        df_train,df_test = load_and_split(data_path,random_state)
        df_train.to_csv(os.path.join(target_data_path,'train.csv'),index=False)
        df_test.to_csv(os.path.join(target_data_path,'test.csv'),index=False)
        print(f'Saved generated data to {target_data_path}')
        print(' ')
        print("******************************************************")
        print('**********  load data - run success   ****************')
        print("******************************************************")
    except Exception as e:
        print('Unable to load the data')
        raise e


    

