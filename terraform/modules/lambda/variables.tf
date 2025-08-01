variable "function_name" {
  description = "The name of the Lambda function."
  type        = string
}

variable "handler" {
  description = "The Lambda function handler (e.g., main.lambda_handler)."
  type        = string
}

variable "runtime" {
  description = "The Lambda function runtime (e.g., python3.9)."
  type        = string
}

variable "lambda_role_arn" {
  description = "The ARN of the IAM role for the Lambda function."
  type        = string
}

variable "ec2_instance_id" {
  description = "The ID of the EC2 instance the Lambda function will start."
}

variable "aws_region" {
  description = "The AWS region where the Lambda function will operate."
  type        = string
}

variable "source_code_path" {
  description = "The local path to the Lambda function's Python script."
  type        = string
}

variable "event_rule_arn" {
  description = "The ARN of the EventBridge rule that will trigger this Lambda."
  type        = string
}
