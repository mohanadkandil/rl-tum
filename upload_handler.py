import json
import boto3
import base64
import os

def handler(event, context):
    s3 = boto3.client('s3')
    dynamodb = boto3.resource('dynamodb').Table('checkers-ai-model-metadata-dev')
    
    try:
        body = json.loads(event['body'])
        model_data = base64.b64decode(body['model'])
        plot_data = base64.b64decode(body['plot'])
        
        # Generate unique checkpoint number
        timestamp = event['requestContext']['requestTime']
        checkpoint_num = str(int(timestamp.replace('-','').replace(':','').replace('T','')))
        
        # Upload to S3
        s3.put_object(
            Bucket='checkers-ai-models-dev',
            Key=f'checkpoints/checkpoint_{checkpoint_num}/model.pth',
            Body=model_data
        )
        s3.put_object(
            Bucket='checkers-ai-models-dev',
            Key=f'checkpoints/checkpoint_{checkpoint_num}/training_plot.png',
            Body=plot_data
        )
        
        # Update DynamoDB
        dynamodb.put_item(Item={
            'model_id': f'checkpoint_{checkpoint_num}',
            'checkpoint_number': checkpoint_num,
            'timestamp': timestamp,
            'model_s3_path': f'checkpoints/checkpoint_{checkpoint_num}/model.pth',
            'plot_s3_path': f'checkpoints/checkpoint_{checkpoint_num}/training_plot.png'
        })
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Upload successful'})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        } 