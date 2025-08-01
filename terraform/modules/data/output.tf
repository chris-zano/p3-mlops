output "mlflow_user_data" {
  value = data.template_file.mlflow_user_data.rendered
}

output "ubuntu_ami_id" {
  value = data.aws_ami_ids.ubuntu_24.ids[0]
}

output "model_train_user_data" {
  value = data.template_file.model_train_user_data.rendered
}

output "lambda_script_path" {
  description = "The path to the Lambda Python script within the module."
  value       = "${path.module}/templates/start_ec2_on_ecr_push.py"
}

output "challenger_lambda_script_path" {
  description = "The path to the Lambda Python script within the module."
  value       = "${path.module}/templates/redeploy_challenger.py"
}