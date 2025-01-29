import boto3
import json
import os
from datetime import datetime
import uuid
from decimal import Decimal

class S3Handler:
    def __init__(self):
        # Set the region for all AWS services
        self.region = 'us-east-1'
        
        # Get today's date for bucket name
        today = datetime.now().strftime('%Y%m%d')
        
        # Initialize AWS clients with correct region
        self.s3 = boto3.client('s3', region_name=self.region)
        self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
        self.cloudfront = boto3.client('cloudfront', region_name=self.region)
        self.sqs = boto3.client('sqs', region_name=self.region)
        
        # Resource names from terraform output
        self.bucket_name = f'checkers-ai-models-dev-{today}'  # Updated bucket name
        self.table = self.dynamodb.Table('checkers-ai-model-metadata-dev')
        self.cloudfront_domain = 'd1x93vkcx4oo18.cloudfront.net'
        self.sqs_queue_url = 'https://sqs.us-east-1.amazonaws.com/558584767754/checkers-ai-training-complete-dev'
        self.checkpoint_counter = 1  # Add counter for checkpoints

    def upload_training(self, checkpoint_path, plot_path, training_info):
        """Upload model and metadata after training"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_num = self.get_next_checkpoint_number()
            model_id = f"checkpoint_{checkpoint_num}"
            
            # Create a unique folder name for this checkpoint
            s3_folder = f"checkpoints/checkpoint_{checkpoint_num}"
            
            # Upload all files from checkpoints directory
            checkpoint_dir = os.path.dirname(checkpoint_path)
            for filename in os.listdir(checkpoint_dir):
                local_file_path = os.path.join(checkpoint_dir, filename)
                s3_key = f"{s3_folder}/{filename}"  # Keep original filename under unique folder
                print(f"Uploading {local_file_path} to {s3_key}")
                self.s3.upload_file(local_file_path, self.bucket_name, s3_key)
            
            # Save metadata to DynamoDB
            metadata = {
                'model_id': model_id,
                'checkpoint_number': checkpoint_num,
                'timestamp': timestamp,
                'win_rate': Decimal(str(training_info['win_rate'])),
                'episodes': training_info['episodes'],
                'model_s3_path': f"{s3_folder}/{os.path.basename(checkpoint_path)}",
                'plot_s3_path': f"{s3_folder}/{os.path.basename(plot_path)}",
                'final_reward': Decimal(str(training_info.get('final_reward', 0))),
                'training_duration': Decimal(str(training_info.get('duration', 0)))
            }
            
            print("Saving metadata to DynamoDB:", metadata)
            response = self.table.put_item(Item=metadata)
            print("DynamoDB response:", response)  # Add this to see the response
            
            # Verify the upload
            self.verify_upload(model_id)
        except Exception as e:
            print(f"Error in upload_training: {str(e)}")
            raise

    def get_next_checkpoint_number(self):
        """Get the next available checkpoint number"""
        try:
            response = self.table.scan()
            items = response['Items']
            if not items:
                return 1
            
            # Get highest checkpoint number and add 1
            max_checkpoint = max(
                [int(item.get('checkpoint_number', 0)) for item in items]
            )
            return max_checkpoint + 1
        except Exception as e:
            print(f"Error getting next checkpoint number: {e}")
            return 1

    def get_available_models(self):
        """Get list of available models with metadata"""
        response = self.table.scan()
        models = response['Items']
        # Sort by win rate
        models.sort(key=lambda x: x['win_rate'], reverse=True)
        return models

    def download_model(self, model_id):
        """Download model from S3"""
        # Get model path from DynamoDB
        response = self.table.get_item(Key={'model_id': model_id})
        s3_path = response['Item']['model_s3_path']
        local_path = os.path.join('checkpoints', os.path.basename(s3_path))
        
        # Create checkpoints directory if it doesn't exist
        os.makedirs('checkpoints', exist_ok=True)
        
        # Download from S3
        self.s3.download_file(self.bucket_name, s3_path, local_path)
        return local_path

    def log_model_metrics(self, model_id, metrics):
        cloudwatch = boto3.client('cloudwatch')
        cloudwatch.put_metric_data(
            Namespace='CheckersAI',
            MetricData=[{
                'MetricName': 'WinRate',
                'Value': metrics['win_rate'],
                'Unit': 'Percent',
                'Dimensions': [{'Name': 'ModelID', 'Value': model_id}]
            }]
        )

    def verify_upload(self, model_id):
        """Verify metadata was saved to DynamoDB"""
        try:
            response = self.table.get_item(
                Key={
                    'model_id': model_id  # Only use model_id as the key
                }
            )
            print("DynamoDB entry:", response['Item'])
            return True
        except Exception as e:
            print(f"Error verifying upload: {e}")
            return False 