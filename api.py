from fastapi import FastAPI
app = FastAPI()

@app.get("/models")
async def list_models():
    storage = ModelStorage()
    return storage.get_available_models()

@app.get("/models/{model_id}/download")
async def get_download_url(model_id: str):
    return generate_presigned_url(model_id) 