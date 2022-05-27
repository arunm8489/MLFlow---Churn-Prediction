import mlflow
from utils import read_config

def main():
    with mlflow.start_run() as run:
        mlflow.run(".", "load_data", use_conda=False)
        mlflow.run(".", "preprocess_data", use_conda=False)
        mlflow.run(".", "tune_params", use_conda=False)
        mlflow.run(".", "train_model", use_conda=False)


if __name__=="__main__":
    try:
        main()
    except Exception as e:
        raise e
