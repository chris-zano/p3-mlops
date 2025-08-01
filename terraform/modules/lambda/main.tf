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
  output_path = "${path.module}/lambda_function.zip"
}

# AWS Lambda Function
resource "aws_lambda_function" "ecr_push_starter" {
  function_name    = var.function_name
  handler          = var.handler
  runtime          = var.runtime
  role             = var.lambda_role_arn
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  timeout          = 30
  memory_size      = 128 

  environment {
    variables = {
      EC2_INSTANCE_ID = var.ec2_instance_id
      APP_REGION      = var.aws_region
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
  function_name = aws_lambda_function.ecr_push_starter.function_name
  principal     = "events.amazonaws.com"
  source_arn    = var.event_rule_arn # This will be passed from the event_bridge module
}
