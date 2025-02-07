from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import Response
from starlette.middleware.cors import CORSMiddleware
from bson import ObjectId
from typing import List
from fastapi.responses import JSONResponse
from fastapi import  HTTPException, File, Form, UploadFile
import os
import database
import uuid
import assistant
import ffmpeg
import aiofiles

def convert_mp4_to_mp3(input_file, output_file):
    try:
        ffmpeg.input(input_file).output(output_file, acodec='libmp3lame', audio_bitrate='192k').run()
        print(f"Conversion successful: {output_file}")
    except ffmpeg.Error as e:
        print(f"Error: {e.stderr.decode()}")
db=database.Database("assistantdb")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, replace with specific domains for more control
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
data_store = []
class Item(BaseModel):
    id: int
    name: str
    description: str

@app.get("/records/")
async def get_items():
    data_list=db.getRecords()
    
    return Response(content=data_list, media_type='application/json')
@app.put("/delete/{item_id}")
async def delete(item_id: str):
    try:
        item_id=ObjectId(item_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID format")
    result=db.deleteRecords(item_id)
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return JSONResponse(content={"message": "Status updated successfully"}, status_code=200)
@app.get("/search")
async def search_items(q: str ):
    data_list=await db.searchRecords(q)    
    return Response(content=data_list, media_type='application/json')
@app.get("/result")
async def result(id: str):
    #Assistant=assistant.Assistant(id)
    result=db.getRecordById(id)
    return Response(content=result, media_type='application/json')


@app.post("/upload/")
async def upload_file(title:str= Form(...),desc:str= Form(...),file:UploadFile = File(...)):
    
    UPLOAD_DIR = "uploads/"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    unique_filename = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    new_filename = f"{unique_filename}{file_extension}"

    
    try:
        
        file_path = os.path.join(UPLOAD_DIR, new_filename)
        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(await file.read())
        

        db.addRecord(title,desc,new_filename)        
        
        return JSONResponse(content={"message": "File uploaded successfully"}, status_code=200)
    
    except Exception as e:
        return JSONResponse(content={"message": f"Error: {str(e)}"}, status_code=400)



