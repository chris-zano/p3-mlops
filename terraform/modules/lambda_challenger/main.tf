terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
    }
  }
}

# Create a zip archive of the Lambda function code
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_file = var.source_code_path
  output_path = "${path.module}/challenger_redeployment.zip"
}

# AWS Lambda Function
resource "aws_lambda_function" "challenger_redeployment_lambda" {
  function_name    = var.function_name
  handler          = "redeploy_challenger.lambda_handler"
  runtime          = "python3.9"
  role             = var.lambda_role_arn
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  timeout          = 60 # Set a generous timeout
  memory_size      = 128

  environment {
    variables = {
      # Pass key configuration as environment variables for the Lambda to use
      ECS_CLUSTER_NAME  = var.ecs_cluster_name
      ECS_SERVICE_NAME  = var.ecs_service_name
      CONTAINER_NAME    = var.container_name
    }
  }

  tags = {
    Name = "${var.function_name}-lambda"
  }
}

# Permission for EventBridge to invoke the Lambda function
resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.challenger_redeployment_lambda.function_name
  principal     = "events.amazonaws.com"
  source_arn    = var.event_rule_arn
}
