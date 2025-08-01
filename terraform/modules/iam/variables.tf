
variable "mlflow_buckets_arns" {
  description = "value for the mlflow_buckets_arns"
  type        = list(string)
}

variable "lambda_role_name" {
  description = "The name for the Lambda execution role."
  type        = string
}

variable "ec2_instance_id" {
  description = "The ID of the EC2 instance the Lambda function will start."
  type        = string
}

variable "aws_region" {
  description = "The AWS region where the EC2 instance and Lambda function are deployed."
  type        = string
}

variable "aws_account_id" {
  description = "The AWS account ID."
  type        = string
}

variable "project_name" {
  
}

variable "ecs_cluster_name" {
  
}

variable "inference_challenger_ecs_service_name" {
  
} 
