# Defines input variables for the EventBridge Challenger module.
variable "rule_name" {
  description = "The name of the EventBridge rule."
  type        = string
}

variable "ecr_repository_name" {
  description = "The name of the ECR repository to monitor for pushes."
  type        = string
}

variable "lambda_function_arn" {
  description = "The ARN of the Lambda function to trigger."
  type        = string
}

# NEW: A variable for the specific ECR image tag to filter on.
variable "ecr_image_tag" {
  description = "The specific ECR image tag to trigger the rule on."
  type        = string
}
