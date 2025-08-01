variable "function_name" {
  description = "The name of the Lambda function."
  type        = string
}

variable "lambda_role_arn" {
  description = "The ARN of the IAM role for the Lambda function."
  type        = string
}

variable "ecs_cluster_name" {
  description = "The name of the ECS cluster."
  type        = string
}

variable "ecs_service_name" {
  description = "The name of the ECS service to update."
  type        = string
}

variable "container_name" {
  description = "The name of the container within the task definition to update."
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
