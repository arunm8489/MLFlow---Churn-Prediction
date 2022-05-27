# MLFlow-Churn-Prediction

Telecom Churn Prediction Project with Experiment tracking and deployment based on Mlflow


**Tools used**
* MLFlow - For experiment tracking
* HyperOpt - For hyperparameter tuning
* Fast Api - Foe deployment




## Stage 1
<p>Suppose we have 3 developers A,B and C working on same project and they are experimenting with different models. In this setup they will log artifacts to an aws S3 bucket and metrics to a mysql db in amazon rds. By this way anyone in team can verify all experiment results.</p>

<img src="https://www.mlflow.org/docs/latest/_images/scenario_4.png">


  <p>First create a virtual environment and activate it from conda shell</p>

```
conda create --prefix ./env python=3.7 -y
conda activate ./env

```
  <p>Now install mlflow using conda</p>

```
conda install -c conda-forge mlflow

```
  <p>Now install all packages in requirements.txt.</p>

```
pip install -r requirements.txt

```

<p>Create a s3 bucket in aws to store artifacts(with public access). Similary create an RDS instance to store metrics and paramters.S3 will become our artifact store and mysql will become our backend store. </p>

<p> Once it is done developers can start experiments. If one would like to view experiments via ui, use the following command </p>

```
mlflow server --backend-store-uri mysql://admin:redhat123@database-mlflow.c4cohmxef4v5.us-east-1.rds.amazonaws.com/mlflowdb --default-artifact-root S3:/artifact-store-bucket001 --host 127.0.0.1 -p 5000

```
Here admin is the mysql username and redhat123 is the password.The enpoint url is database-mlflow.c4cohmxef4v5.us-east-1.rds.amazonaws.com and db name is mlflowdb.
similarly artifact root is S3:/artifact-store-bucket001. Here we are using local tracking server(127.0.0.1) with port 5000.(You can find the code inside "Experiments" folder)


Here after feature engineering as modeling I have tried logistic regression and RandomForest. Once we have base line model. We can push that model to staging and then in to production (It completly depends on team. We can do that if we need a faster development cycle).

## Stage 2

Suppose the developer need to tune the models. Here, we used a small dataset, but assume our dataset is large. In that case we can easily tune Logistic regresion in local computer itselt since train complexity of logistc regression is small - O(d) where d is the dimension. But in the case of Random Forest we have train complexity of  O(ndk) where k->no of trees, d-> depth of tree,n-> no of datapoints. So its better if we can package the code and move it to cloud for tuning. In that case we can use MLFlow projects.(You can find the code inside "MLFlow-Project" folder)


**MLflow Projects it is an MLflow format/convention for packaging Machine Learning code in a reusable and reproducible way. It allows a Machine Learning code to be decomposed into small chunks that address very specific use cases (e.g. data loading/processing, model training, etc.) and then chaining them together to form the final machine learning workflow.**

You can find the entire pipeline including hyperparameter tuning inside the MLProject directory.

Inorder to run the full pipeline

```
mlflow run . -e train_pipeline
```

## Stage 3

<img src="https://miro.medium.com/max/841/1*qYW_eNpc_1bir6MRdZtW2g.png">


Once we have the best model after finetuning. We can mark the model as production and develop and endpoint for deployment. We have use fast api fpr api development.(You can find the code inside "api" folder)
