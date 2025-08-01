import boto3
import os
import json

# Define your EC2 instance ID and region
# It's best practice to pass these as environment variables to the Lambda function
# or retrieve them from the event if the event contains enough context.
# For simplicity, we'll use environment variables here.
EC2_INSTANCE_ID = os.environ.get('EC2_INSTANCE_ID')
AWS_REGION = os.environ.get('APP_REGION')

# Initialize the EC2 client
ec2 = boto3.client('ec2', region_name=AWS_REGION)

def lambda_handler(event, context):
    """
    Lambda function to start a specific EC2 instance upon an ECR image push event.
    """
    print(f"Received event: {json.dumps(event, indent=2)}")

    # Basic validation for the EC2_INSTANCE_ID
    if not EC2_INSTANCE_ID:
        print("Error: EC2_INSTANCE_ID environment variable is not set.")
        return {
            'statusCode': 400,
            'body': json.dumps('EC2_INSTANCE_ID environment variable not set.')
        }

    try:
        # Describe instances to check current state (optional, but good for robust logic)
        # You might want to only start if it's currently stopped.
        response = ec2.describe_instances(InstanceIds=[EC2_INSTANCE_ID])
        reservations = response['Reservations']
        current_state = None
        if reservations:
            instance = reservations[0]['Instances'][0]
            current_state = instance['State']['Name']
            print(f"EC2 instance {EC2_INSTANCE_ID} is currently in state: {current_state}")

        if current_state == 'stopped':
            # Start the EC2 instance
            print(f"Attempting to start EC2 instance: {EC2_INSTANCE_ID} in region {AWS_REGION}")
            start_response = ec2.start_instances(InstanceIds=[EC2_INSTANCE_ID])
            
            # Log the response from starting the instance
            for instance in start_response['StartingInstances']:
                print(f"Instance {instance['InstanceId']} state changed from {instance['PreviousState']['Name']} to {instance['CurrentState']['Name']}")
            
            return {
                'statusCode': 200,
                'body': json.dumps(f'Successfully started EC2 instance: {EC2_INSTANCE_ID}')
            }
        elif current_state == 'running':
            print(f"EC2 instance {EC2_INSTANCE_ID} is already running. No action needed.")
            return {
                'statusCode': 200,
                'body': json.dumps(f'EC2 instance {EC2_INSTANCE_ID} already running.')
            }
        else:
            print(f"EC2 instance {EC2_INSTANCE_ID} is in state '{current_state}'. Not attempting to start.")
            return {
                'statusCode': 200,
                'body': json.dumps(f'EC2 instance {EC2_INSTANCE_ID} in unexpected state: {current_state}. No action taken.')
            }

    except Exception as e:
        print(f"Error starting EC2 instance {EC2_INSTANCE_ID}: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error starting EC2 instance: {e}')
        }
