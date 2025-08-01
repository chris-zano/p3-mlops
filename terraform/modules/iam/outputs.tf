output "mlflow_instance_profile_name" {
  value = aws_iam_instance_profile.mlflow_instance_profile.name
}

output "ecr_access_iam_role" {
  value = aws_iam_role.ecr_access.arn
}

output "model_training_role" {
  value = aws_iam_instance_profile.model_training_profile.name
}

output "lambda_ec2_starter_role_arn" {
  description = "The ARN of the IAM role for the Lambda function."
  value       = aws_iam_role.lambda_ec2_starter_role.arn
}

# Add this output to your existing modules/iam/outputs.tf

output "challenger_redeployment_role_arn" {
  description = "The ARN of the IAM role for the challenger redeployment Lambda function."
  value       = aws_iam_role.challenger_redeployment_role.arn
}
