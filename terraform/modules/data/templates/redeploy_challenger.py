import boto3
import os
import json

# Initialize ECS and ECR clients
ecs_client = boto3.client('ecs')

def lambda_handler(event, context):
    """
    Redeploys the ECS Challenger service with the new Docker image from an ECR push event.
    The new model version is passed as an environment variable to the container.
    """
    print(f"Received event: {json.dumps(event, indent=2)}")

    try:
        # Get environment variables from the Lambda configuration
        ecs_cluster_name = os.environ.get('ECS_CLUSTER_NAME')
        ecs_service_name = os.environ.get('ECS_SERVICE_NAME')
        container_name = os.environ.get('CONTAINER_NAME')

        if not all([ecs_cluster_name, ecs_service_name, container_name]):
            raise ValueError("Required environment variables not set.")

        # Extract image URI and tag from the ECR event
        detail = event['detail']
        repository_name = detail['repository-name']
        image_tag = detail['image-tag']
        image_uri = f"{detail['repository-name']}:{image_tag}"

        print(f"New image pushed to ECR: {image_uri}")

        # Describe the current service to get its task definition ARN
        describe_service_response = ecs_client.describe_services(
            cluster=ecs_cluster_name,
            services=[ecs_service_name]
        )
        service = describe_service_response['services'][0]
        current_task_definition_arn = service['taskDefinition']
        print(f"Current Task Definition ARN: {current_task_definition_arn}")

        # Describe the current task definition to get the container definitions
        describe_td_response = ecs_client.describe_task_definition(
            taskDefinition=current_task_definition_arn
        )
        task_definition = describe_td_response['taskDefinition']
        container_definitions = task_definition['containerDefinitions']
        
        # Find the specific container and update its image URI
        new_container_definitions = []
        for container in container_definitions:
            if container['name'] == container_name:
                print(f"Updating container '{container_name}' with new image: {image_uri}")
                container['image'] = image_uri

                # Find or add the MODEL_VERSION environment variable
                env_vars = container.get('environment', [])
                model_version_found = False
                for env_var in env_vars:
                    if env_var['name'] == 'MODEL_VERSION':
                        env_var['value'] = image_tag
                        model_version_found = True
                        print(f"Updated MODEL_VERSION to: {image_tag}")
                        break
                if not model_version_found:
                    env_vars.append({'name': 'MODEL_VERSION', 'value': image_tag})
                    container['environment'] = env_vars
                    print(f"Added MODEL_VERSION with value: {image_tag}")
                
            new_container_definitions.append(container)

        # Register a new task definition with the updated image
        register_td_response = ecs_client.register_task_definition(
            family=task_definition['family'],
            containerDefinitions=new_container_definitions,
            taskRoleArn=task_definition.get('taskRoleArn'),
            executionRoleArn=task_definition.get('executionRoleArn'),
            networkMode=task_definition.get('networkMode'),
            requiresCompatibilities=task_definition['requiresCompatibilities'],
            cpu=task_definition['cpu'],
            memory=task_definition['memory']
        )
        new_task_definition_arn = register_td_response['taskDefinition']['taskDefinitionArn']
        print(f"Registered new Task Definition: {new_task_definition_arn}")
        
        # Update the ECS service to use the new task definition
        update_service_response = ecs_client.update_service(
            cluster=ecs_cluster_name,
            service=ecs_service_name,
            taskDefinition=new_task_definition_arn
        )
        print(f"Successfully triggered redeployment of service {ecs_service_name}")

        return {
            'statusCode': 200,
            'body': json.dumps('ECS service update triggered successfully.')
        }

    except Exception as e:
        print(f"Error during challenger redeployment: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {e}')
        }
