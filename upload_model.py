import requests
import json
import os

def upload_model(model_path, plot_path, api_key=None):
    url = "https://uwwq7nwgg6.execute-api.us-east-1.amazonaws.com/dev/models"
    
    api_key = api_key or os.getenv('CHECKERS_AI_API_KEY')
    if not api_key:
        raise ValueError("API key required. Set CHECKERS_AI_API_KEY environment variable.")
    
    headers = {
        'x-api-key': api_key
    }
    
    files = {
        'model': open(model_path, 'rb'),
        'plot': open(plot_path, 'rb')
    }
    
    response = requests.post(url, headers=headers, files=files)
    return response.json()

if __name__ == "__main__":
    # Test the upload
    result = upload_model(
        'checkpoints/best_model.pth',
        'checkpoints/training_results.png'
    )
    print(result) 