import requests
import json
import os

def upload_model(checkpoint_dir, checkpoint_number=0, episodes=0, final_reward=0, 
                training_duration=0, win_rate=0, api_key=None):
    url = "https://uwwq7nwgg6.execute-api.us-east-1.amazonaws.com/dev/models/presigned-url"
    
    api_key = api_key or os.getenv('CHECKERS_AI_API_KEY')
    if not api_key:
        raise ValueError("API key required. Set CHECKERS_AI_API_KEY environment variable.")
    
    print(f"Using API key: {api_key[:5]}...{api_key[-5:]}")
    
    try:
        # Get list of files to upload
        files_to_upload = []
        for filename in os.listdir(checkpoint_dir):
            if filename.endswith('.pth') or filename.endswith('.png'):
                files_to_upload.append(filename)
        
        # Get presigned URLs for all files
        print("Getting presigned URLs...")
        params = {
            'files': ','.join(files_to_upload),
            'checkpoint_number': str(checkpoint_number),
            'episodes': str(episodes),
            'final_reward': str(final_reward),
            'training_duration': str(training_duration),
            'win_rate': str(win_rate)
        }
        response = requests.get(url, headers={'x-api-key': api_key}, params=params)
        response.raise_for_status()
        data = response.json()
        
        print(f"Got URLs for folder: {data['folder']}")
        
        # Upload each file
        for filename, presigned_url in data['urls'].items():
            filepath = os.path.join(checkpoint_dir, filename)
            print(f"Uploading {filename}...")
            with open(filepath, 'rb') as f:
                response = requests.put(presigned_url, data=f)
                response.raise_for_status()
            print(f"Uploaded {filename}")
        
        return {"message": "Upload successful", "folder": data['folder']}
            
    except requests.exceptions.RequestException as e:
        print(f"Upload failed: {e}")
        if hasattr(e, 'response'):
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        raise

if __name__ == "__main__":
    # Test the upload
    result = upload_model(
        'checkpoints'
    )
    print(result) 