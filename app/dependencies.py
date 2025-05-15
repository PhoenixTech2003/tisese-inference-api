from typing import Annotated
from fastapi import Depends, UploadFile, HTTPException
import json
import requests
import os
import cv2
import tempfile
from supabase import create_client, Client





async def run_inference(file: UploadFile):
    try:
        # Validate API key and required environment variables
        api_key = os.getenv("ULTRALYTICS_API_KEY")
        model_url = os.getenv("ULTRALYTICS_MODEL_URL")
        inference_url = os.getenv("ULTRALYTICS_INFERENCE_URL")
        
        if not all([api_key, model_url, inference_url]):
            raise HTTPException(status_code=500, detail="Missing required environment variables")
        
        # Set up request parameters
        headers = {"x-api-key": api_key}
        data = {"model": model_url, "imgsz": 640, "conf": 0.25, "iou": 0.45}
        
        # Read the file content
        try:
            file_content = await file.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
        
        # Validate file content
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file provided")
        
        # Properly format the file for the API request
        files = {
            "file": (file.filename, file_content, file.content_type)
        }
        
        # Make the API request
        try:
            response = requests.post(
                inference_url,
                headers=headers,
                data=data,
                files=files
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if response := getattr(e, 'response', None):
                try:
                    error_detail = response.json()
                    error_msg = f"{error_msg}: {json.dumps(error_detail)}"
                except (ValueError, json.JSONDecodeError):
                    if response.text:
                        error_msg = f"{error_msg}: {response.text}"
            raise HTTPException(status_code=500, detail=f"Ultralytics API error: {error_msg}")
        
        # Parse the response
        try:
            result = response.json() 
        except (ValueError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Error parsing API response: {str(e)}")
        try:
            # Save the file content to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file.filename.split(".")[-1]}') as temp_file:
                temp_file_path = temp_file.name
                # Reset the file pointer to the beginning
                await file.seek(0)
                # Read the file content again
                file_content = await file.read()
                # Write to the temporary file
                temp_file.write(file_content)
            
            # Read the image with OpenCV
            image = cv2.imread(temp_file_path)
            if image is None:
                raise Exception(f"Failed to read image from {temp_file_path}")
            
            # Check if the result contains the expected data structure
            if not result.get("images") or len(result["images"]) == 0:
                raise Exception("No images found in the result")
                
            if not result["images"][0].get("results") or len(result["images"][0]["results"]) == 0:
                # No detection results, return the original image
                print("No detection results found in the API response")
                # Clean up the temporary file
                os.unlink(temp_file_path)
                
                # Convert the image to bytes directly without saving to disk
                _, buffer = cv2.imencode('.jpg', image)
                image_bytes = buffer.tobytes()
                
                return {
                    "original_file": file,
                    "image_bytes": image_bytes,
                    "filename": file.filename
                }
                
            # Get bounding box coordinates
            box = result["images"][0]["results"][0].get("box", {})
            if not box or not all(k in box for k in ["x1", "y1", "x2", "y2"]):
                # Missing box coordinates, return the original image
                print("Missing box coordinates in the API response")
                # Clean up the temporary file
                os.unlink(temp_file_path)
                
                # Convert the image to bytes directly without saving to disk
                _, buffer = cv2.imencode('.jpg', image)
                image_bytes = buffer.tobytes()
                
                return {
                    "original_file": file,
                    "image_bytes": image_bytes,
                    "filename": file.filename
                }
            
            x1 = int(box["x1"])
            y1 = int(box["y1"])
            x2 = int(box["x2"])
            y2 = int(box["y2"])
            
            # Draw rectangle on image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
            # Convert the image to bytes directly without saving to disk
            # Encode the image to the appropriate format (JPEG)
            _, buffer = cv2.imencode('.jpg', image)
            image_bytes = buffer.tobytes()
            
            # Return the image bytes and filename for use in save_to_supabase_storage
            return {
                "original_file": file,
                "image_bytes": image_bytes,
                "filename": file.filename
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Unexpected error in run_inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    
def SupabaseClient() -> Client:
    try:
        url: str = os.getenv("SUPABASE_URL")
        key: str = os.getenv("SUPABASE_KEY")
        supabase: Client = create_client(url, key)    
        return supabase
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading Supabase client: {str(e)}")
    
async def save_to_supabase_storage(inference_result: Annotated[dict, Depends(run_inference)], supabase: Client = Depends(SupabaseClient)):
    try:
        # Get the image bytes and filename from the inference result
        image_bytes = inference_result["image_bytes"]
        filename = inference_result["filename"]
        
        # Upload the processed image bytes directly to Supabase
        bucket_name = os.getenv("SUPABASE_STORAGE_BUCKET")
        if not bucket_name:
            raise HTTPException(status_code=500, detail="SUPABASE_STORAGE_BUCKET environment variable not set")
            
        response = (
            supabase.storage
            .from_(bucket_name)
            .upload(
                path=f"inference/output_{filename}",
                file=image_bytes,
                file_options={"cache-control": "3600", "upsert": "true"}
            )
        )
        
        # Get the public URL for the uploaded file
        file_url = supabase.storage.from_(bucket_name).get_public_url(f"inference/output_{filename}")
        
        return file_url
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving image to Supabase: {str(e)}")