from typing import Annotated
from fastapi import Depends, UploadFile, HTTPException
import json
import requests
import os
import cv2
import tempfile

async def run_inference(file: UploadFile) -> str:
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
            print(result["images"][0]["results"][0]["box"]["x1"])
            print(json.dumps(result, indent=2)) 
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
                
            # Get bounding box coordinates
            x1 = int(result["images"][0]["results"][0]["box"]["x1"])
            y1 = int(result["images"][0]["results"][0]["box"]["y1"])
            x2 = int(result["images"][0]["results"][0]["box"]["x2"])
            y2 = int(result["images"][0]["results"][0]["box"]["y2"])
            
            # Draw rectangle on image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Save the image with bounding box
            output_path = f"output_{file.filename}"
            cv2.imwrite(output_path, image)
            
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
            # For server environments, we don't use imshow as it requires a GUI
            # Instead, we save the image and return its path
            print(f"Image with bounding box saved to {output_path}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
        # Return the results URL from the response if available, otherwise return filename
        return image
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Unexpected error in run_inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    
    
    
async def save_to_supabase_storage(inference_image:Annotated[UploadFile, Depends(run_inference)]):
    try:
        
        print()    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving image to Supabase: {str(e)}")