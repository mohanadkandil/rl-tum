import json
import boto3
import uuid
import os

def handler(event, context):
    s3 = boto3.client('s3')
    bucket = 'checkers-ai-models-dev-20250129'
    
    try:
        upload_id = str(uuid.uuid4())[:8]
        base_folder = f"checkpoints/checkpoint_{upload_id}"
        
        files = event.get('queryStringParameters', {}).get('files', '').split(',')
        if not files:
            files = ['model.pth', 'plot.png', 'model_final_e100.pth']  
        
        urls = {}
        for filename in files:
            key = f"{base_folder}/{filename}"
            url = s3.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': bucket,
                    'Key': key
                },
                ExpiresIn=3600
            )
            urls[filename] = url
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'urls': urls,
                'folder': base_folder
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }