from sklearn.metrics import confusion_matrix,roc_auc_score,auc,roc_curve,classification_report
import argparse,os
import mlflow
import pandas as pd
from utils import read_config,plot_confusion_matrixes
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

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


def train_rf_model(run_name,parameters,artifact_path):
    with mlflow.start_run(run_name=run_name) as run:
        print(f'Run id: {run.info.run_uuid}')
        print(f'Run name: {run_name}')
        print(f'Exp id: {run.info.experiment_id}')
        
        parameters["random_state"] = 42
        model = RandomForestClassifier(**parameters)
        mlflow.log_params(parameters)
        model.fit(X_train,y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        #cross validation
        # do cross validation to get cv auc
        # cv_auc = stratified_cross_model(X_train,y_train)

        train_fpr, train_tpr, thresholds = roc_curve(y_train, model.predict_proba(X_train)[:,1])
        test_fpr, test_tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    
        train_auc = auc(train_fpr, train_tpr)
        test_auc = auc(test_fpr, test_tpr)

        #Area under ROC curve
        print('Area under train roc {}'.format(train_auc))
        # print('Area under cv roc {}'.format(cv_auc))
        print('Area under test roc {}'.format(test_auc))
        mlflow.log_metrics({'train_auc':train_auc,'test_auc':test_auc})
        model_path = os.path.join(artifact_path,"model")
        print(f'Saving model to {model_path}')
        mlflow.sklearn.save_model(model, model_path)

        print('Saving ROC Curve...')
        plt.figure(figsize=(10,5))
        sns.set(font_scale = 2)
        plt.grid(True)
        plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
        plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
        plt.legend()
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        plt.title("ROC CURVE FOR OPTIMAL K")
        fig_path = os.path.join(artifact_path,"ROCcurve.png")
        plt.savefig(fig_path)
        mlflow.log_artifact(fig_path)
        plt.show()
        # os.remove("ROCcurve.png")
        print('Saving confusion matrix....')
        plot_confusion_matrixes(y_train,y_train_pred,y_test,y_test_pred,artifact_path)
        train_report = classification_report(y_train,y_train_pred)
        print(f'Train classification report: ')
        print(train_report)
        test_report = classification_report(y_test,y_test_pred)
        print(f'Test classification report: ')
        print(test_report)



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()
    conf_dict = read_config(parsed_args.config)

    cat_save_path = conf_dict['config']['artifacts']['cat_col_dict_path']
    scalar_save_path = conf_dict['config']['artifacts']['std_scalar_path']
    data_dir_path = conf_dict['config']['artifacts']['data_dir_path']
    artifact_path = conf_dict['config']['artifacts']['dir_path']

    params = conf_dict['params']
    
    parameters = { 
        'criterion':params["criterion"],
        'max_depth':params["max_depth"],
        'n_estimators':params["n_estimators"],
        'min_samples_split':params["min_samples_split"],
        'min_samples_leaf':params["min_samples_leaf"]
    }

    try:
        X_train,y_train,X_test,y_test = load_train_test(data_dir_path)
        print('Train and test processed data loaded')
    except Exception as e:
        print('Unable to load the processed dara')
        raise e

    try:
        print('Training model ...')
        train_rf_model('random_forest_final',parameters,artifact_path)
        print(' ')
        print("******************************************************")
        print('*********  train model - run success   ***************')
        print("******************************************************")
    except Exception as e:
        print('Unable to train the model')
        print(e)


    
