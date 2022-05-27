import yaml 
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os,mlflow
import pandas as pd
import matplotlib.pyplot as plt

def read_config(config_path):
    with open(config_path) as cfg_file:
        content = yaml.safe_load(cfg_file)
    return content

def write_params(config_path,best_params):
    with open(config_path) as cfg_file:
        content = yaml.safe_load(cfg_file)

    content["params"]["criterion"] = best_params["criterion"]
    content["params"]["max_depth"] = best_params["max_depth"]
    content["params"]["n_estimators"] = best_params["n_estimators"]
    content["params"]["min_samples_split"] = best_params["min_samples_split"]
    content["params"]["min_samples_leaf"] = best_params["min_samples_leaf"]

    with open(config_path, 'w') as c:
        yaml.dump(content , c)

    

# helper function to plot confusion matrix
def plot_confusion_matrixes(y_train,y_train_pred,y_test,y_test_pred,artifact_path):
    
    sns.set(font_scale = 1)
    cm_train = confusion_matrix(y_train,y_train_pred)
    cm_test =  confusion_matrix(y_test,y_test_pred)
    class_label = ["Not churn","Churn",]
    df_train = pd.DataFrame(cm_train, index = class_label, columns = class_label)
    df_test = pd.DataFrame(cm_test, index = class_label, columns = class_label)
    f, axes = plt.subplots(1, 2,figsize=(12,4))
    fig_path = os.path.join(artifact_path,"confusion_matrix.png")
    for i in range(2):
      df = df_train if i==0 else df_test
      sns.heatmap(df, annot = True, fmt = "d",ax=axes[i])
      axes[i].set_title(f"Confusion Matrix - {'Train' if i==0 else 'Test'}")
      axes[i].set_xlabel("Predicted Label")
      axes[i].set_ylabel("True Label")
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)
    plt.show()