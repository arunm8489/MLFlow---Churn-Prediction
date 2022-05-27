# MLFlow-Churn-Prediction

Telecom Churn Prediction Project with Experiment tracking and deployment based on Mlflow


**Tools used**
* MLFlow - For experiment tracking
* HyperOpt - For hyperparameter tuning
* Fast Api - Foe deployment


**Scenario**

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
#mlflow server --backend-store-uri mysql://admin:redhat123@database-mlflow.c4cohmxef4v5.us-east-1.rds.amazonaws.com/mlflowdb --default-artifact-root S3:/artifact-store-bucket001 --host 127.0.0.1 -p 5000

```
Here admin is the mysql username and redhat123 is the password.The enpoint url is database-mlflow.c4cohmxef4v5.us-east-1.rds.amazonaws.com and db name is mlflowdb.
similarly artifact root is S3:/artifact-store-bucket001. Here we are using local tracking server(127.0.0.1) with port 5000

