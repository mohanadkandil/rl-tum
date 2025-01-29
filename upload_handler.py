import json
import boto3
import base64
from datetime import datetime

def handler(event, context):
    s3 = boto3.client('s3')
    bucket = 'checkers-ai-models-dev-20250129'
    
    try:
        # Get files from multipart form
        body = event['body']
        
        # Generate folder name
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        folder = f"checkpoints/checkpoint_{timestamp}"
        
        # Upload files to S3
        s3.put_object(
            Bucket=bucket,
            Key=f"{folder}/model.pth",
            Body=body['model']
        )
        
        s3.put_object(
            Bucket=bucket,
            Key=f"{folder}/plot.png",
            Body=body['plot']
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Upload successful'})
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        } 