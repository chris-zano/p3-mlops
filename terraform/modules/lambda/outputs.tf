output "lambda_function_arn" {
  description = "The ARN of the Lambda function."
  value       = aws_lambda_function.ecr_push_starter.arn
}
