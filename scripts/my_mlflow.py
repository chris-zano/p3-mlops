import mlflow

mlflow.set_tracking_uri("http://3.248.185.241/")

with mlflow.start_run():
    mlflow.log_param("param1", "value1")
    mlflow.log_metric("metric1", 0.89)