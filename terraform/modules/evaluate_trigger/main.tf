terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
    }
  }
}

# --- Variables ---
variable "stop_instance_id" {
  description = "The ID of the EC2 instance to monitor for a 'stopped' state."
  type        = string
}

variable "start_instance_id" {
  description = "The ID of the EC2 instance to start when the event is triggered."
  type        = string
}

variable "lambda_function_name" {
  description = "The name for the Lambda function."
  type        = string
  default     = "ec2-stopped-handler"
}

# --- Lambda Function ---

# IAM Role for the Lambda function
# This role grants the Lambda function permission to log to CloudWatch and start EC2 instances.
resource "aws_iam_role" "lambda_role" {
  name = "${var.lambda_function_name}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      },
    ]
  })
}

# IAM Policy for the Lambda role
# Grants permissions for CloudWatch Logs and to start the *specific* EC2 instance.
resource "aws_iam_role_policy" "lambda_policy" {
  name = "lambda_policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Effect   = "Allow"
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        "Action": [
                "ec2:DescribeInstances",
                "ec2:StartInstances",
                "ec2:StopInstances",
                "ec2:RebootInstances"
            ],
            "Effect": "Allow",
            "Resource": "*"
      },
      {
        Action = [
          "ec2:StartInstances"
        ]
        Effect = "Allow"
        Resource = "arn:aws:ec2:*:*:instance/${var.start_instance_id}"
      },
    ]
  })
}

# The Lambda function's source code
# A Python function that starts the EC2 instance specified in the environment variables.
resource "aws_lambda_function" "ec2_stopped_lambda" {
  filename         = data.archive_file.lambda_zip.output_path
  function_name    = var.lambda_function_name
  role             = aws_iam_role.lambda_role.arn
  handler          = "lambda_function.lambda_handler"
  runtime          = "python3.9"

  # Pass the ID of the instance to start as an environment variable
  environment {
    variables = {
      START_INSTANCE_ID = var.start_instance_id
    }
  }

  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
}

# Use the archive_file data source to package the Lambda function code
# This is a much more robust approach than using a local-exec provisioner.
data "archive_file" "lambda_zip" {
  type        = "zip"
  output_path = "lambda_function_payload.zip"
  source_content = <<EOF
import boto3
import json
import os

def lambda_handler(event, context):
    print("Received EC2 instance stopped event:")
    print(json.dumps(event, indent=2))
    
    # Get the instance ID to start from the environment variable
    instance_id_to_start = os.environ['START_INSTANCE_ID']
    print(f"Attempting to start instance: {instance_id_to_start}")
    
    try:
        ec2_client = boto3.client('ec2')
        response = ec2_client.start_instances(
            InstanceIds=[instance_id_to_start]
        )
        print(f"Successfully triggered start for instance {instance_id_to_start}. Response: {json.dumps(response, indent=2)}")
        return {
            'statusCode': 200,
            'body': json.dumps(f'Successfully started instance {instance_id_to_start}')
        }
    except Exception as e:
        print(f"Error starting instance {instance_id_to_start}: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error starting instance {instance_id_to_start}: {str(e)}')
        }
EOF
  source_content_filename = "lambda_function.py"
}

# --- EventBridge Rule and Target ---

# EventBridge rule to listen for the EC2 state change
resource "aws_cloudwatch_event_rule" "ec2_stopped_rule" {
  name        = "${var.lambda_function_name}-rule"
  description = "Fires when EC2 instance ${var.stop_instance_id} has stopped."

  event_pattern = jsonencode({
    "source": ["aws.ec2"],
    "detail-type": ["EC2 Instance State-change Notification"],
    "detail": {
      "instance-id": [var.stop_instance_id],
      "state": ["stopped"]
    }
  })
}

# EventBridge target to link the rule to the Lambda function
resource "aws_cloudwatch_event_target" "ec2_stopped_target" {
  rule      = aws_cloudwatch_event_rule.ec2_stopped_rule.name
  target_id = "InvokeLambda"
  arn       = aws_lambda_function.ec2_stopped_lambda.arn
}

# --- Permissions ---

# Permission for EventBridge to invoke the Lambda function
resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ec2_stopped_lambda.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.ec2_stopped_rule.arn
}
