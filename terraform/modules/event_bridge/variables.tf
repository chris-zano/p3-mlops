variable "rule_name" {
  description = "The name of the EventBridge rule."
  type        = string
}

variable "ecr_repository_name" {
  description = "The name of the ECR repository to monitor for pushes."
}

variable "lambda_function_arn" {
  description = "The ARN of the Lambda function to invoke."
  type        = string
}