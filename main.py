from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from app_gemini import extract_receipt_data
import uvicorn
import os

app = FastAPI(title="Receipt Engine API")

# Define your secret key (store this in Replit Secrets/Environment)
API_KEY = os.environ.get("MY_SECRET_API_KEY", "default-key-for-testing")

async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized: Invalid API Key")
    return x_api_key

@app.get("/")
async def root():
    return {"message": "Receipt Engine API is active."}

@app.post("/extract")
async def extract(
    file: UploadFile = File(...), 
    api_key: str = Depends(verify_api_key)
):
    """
    Endpoint to upload a receipt image and get structured JSON.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Read file bytes
        image_bytes = await file.read()
        
        # Call the Gemini extractor
        result = extract_receipt_data(image_bytes)
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to extract data from image.")
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
