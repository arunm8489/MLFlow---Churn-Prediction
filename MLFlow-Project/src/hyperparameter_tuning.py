from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,roc_curve,auc
import argparse,os
import mlflow
import numpy as np
import pandas as pd
from utils import read_config,write_params
from hyperopt import fmin, tpe, hp,Trials,STATUS_OK
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc,roc_auc_score

def stratified_cv(model,X,y,n_folds=5):
    # X = X.reset_index(drop=True)
    # y = y.reset_index(drop=True)
    train_auc,cv_auc = [],[]
    kf = StratifiedKFold(n_splits=n_folds)
    for f, (train_index, val_index) in enumerate(kf.split(X=X, y=y)):
        x_train_, x_cv_ = X.iloc[train_index], X.iloc[val_index]
        y_train_, y_cv_ = y.iloc[train_index], y.iloc[val_index]
        # model = LogisticRegression(**params)
        model.fit(x_train_,y_train_)

        train_fpr, train_tpr, thresholds = roc_curve(y_train_, model.predict_proba(x_train_)[:,1])
        cv_fpr, cv_tpr, thresholds = roc_curve(y_cv_, model.predict_proba(x_cv_)[:,1])
        train_auc_ = auc(train_fpr, train_tpr)
        cv_auc_ = auc(cv_fpr, cv_tpr)
        train_auc.append(train_auc_)
        cv_auc.append(cv_auc_)
    return np.array(train_auc),np.array(cv_auc)


def objective_fn(params,n_folds=5):

    with mlflow.start_run(nested=True):
        clf = RandomForestClassifier(**params)
        _,auc = stratified_cv(model=clf,X=X_train,y=y_train,n_folds=5)
        mean_auc = auc.mean()
        mlflow.log_params(params)
        mlflow.log_params({"cv_n_fold":n_folds})
        mlflow.log_metrics({"cv_auc":mean_auc})

        # print(f'For params {params} cross validation AUC: {mean_auc}')
        loss = -1*mean_auc
        return {"loss": loss, "status": STATUS_OK}

def load_train_test(data_dir_path):
    train_path = os.path.join(data_dir_path,'train_processed.csv')
    test_path = os.path.join(data_dir_path,'test_processed.csv')
    df_train,df_test = pd.read_csv(train_path),pd.read_csv(test_path)
    X_train = df_train.drop(columns=['Churn'])
    X_test = df_test.drop(columns=['Churn'])
    y_train = df_train['Churn'].reset_index(drop=True)
    y_test = df_test['Churn'].reset_index(drop=True)
    print('Generated data:')
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    
    return X_train,y_train,X_test,y_test


if __name__=="__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()
    conf_dict = read_config(parsed_args.config)

    params = conf_dict["hyperparameter_tuning"]["params"]
    
    print(f'Loaded parameters: {params}')
    criterion = params["criterion"]
    max_depth = params["max_depth"]
    n_estimators = params["n_estimators"]
    min_samples_split = params["min_samples_split"]
    min_samples_leaf = params["min_samples_leaf"]
    data_dir_path = conf_dict['config']['artifacts']['data_dir_path']

    try:
        X_train,y_train,X_test,y_test = load_train_test(data_dir_path)
        print('Train and test processed data loaded')
    except Exception as e:
        print('Unable to load the processed dara')
        raise e
 
    # 1. Define search space
    search_space = {
    "n_estimators": hp.choice("n_estimators", n_estimators),
    "max_depth": hp.choice("max_depth", max_depth),
    "criterion": hp.choice("criterion", criterion),
    "min_samples_split": hp.choice("min_samples_split",min_samples_split),
    "min_samples_leaf": hp.choice("min_samples_leaf",min_samples_leaf)
    }

    trials = Trials()

    with mlflow.start_run(run_name="hyper parameter tuning RF") as run:
        best = fmin(fn=objective_fn,
                space=search_space,
                algo=tpe.suggest,
                max_evals=10,
                trials=trials)
        
        
        best = {k:params[k][v] for k,v in best.items()}
        print(f'Best params: {best}')
        mlflow.log_params(best)
        write_params(parsed_args.config,best)
        


    print("******************************************************")
    print('***********  parameter tuning - run success  *********')
    print("******************************************************")






      

