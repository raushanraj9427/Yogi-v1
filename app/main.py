from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from random import randint
import uuid
from .models import Plants , Details
from sqlalchemy.orm import Session
from .db import  get_db
from fastapi_sqlalchemy import DBSessionMiddleware
import os
load_dotenv(".env")
app = FastAPI()
# app.add_middleware(DBSessionMiddleware, db_url = os.environment["DATABASE_URL"])

IMAGEDIR = "images/"

class DetailBase(BaseModel):
    plant_family: str
    plant_bio: str
    plant_descr: str
    plant_url: str
    
    class config:
        orm_mode = True

class PlantBase(BaseModel):
    plant_text: str
    choices: List[DetailBase]
    
    class config:
        orm_mode = True
        
class Plant(BaseModel):
    plant_text:str
    
    class config:
        orm_mode =True



@app.get("/plants/{plant_id}")
async def read_plant(plant_id: int, db: Session = Depends(get_db)):
    if (
        result := db.query(Plants)
        .filter(Plants.id == plant_id)
        .first()
    ):
        return result
    else:
        raise HTTPException(status_code=404, detail='Plant not found')

@app.get("/details/(plant_id)")
async def read_detail(plant_id:int, db: Session = Depends(get_db)):
    if (
        result := db.query(Details)
        .filter(Details.plant_id == plant_id)
        .first()
    ):
        return result
    else:
        raise HTTPException(status_code=404, detail='Details not found')


@app.post("/plants")
async def create_plant(plant_text:Plant, db: Session = Depends(get_db)):
    db_plant = Plants(**plant_text.dict())

    db.add(db_plant)
    db.commit()
    db.refresh(db_plant)
    return plant_text



@app.post("/plants_details")
async def create_plant_details(plant: PlantBase, db: Session = Depends(get_db)):
    db_plant = Plants(plant_text = plant.plant_text)
    db.add(db_plant)
    db.commit()
    db.refresh(db_plant)
    for choice in plant.choices:
        db_choice = Details(plant_family=choice.plant_family, plant_bio=choice.plant_bio, plant_descr=choice.plant_descr, plant_url= choice.plant_url, plant_id=db_plant.id)
        db.add(db_choice)
    db.commit()


# to upload and load image
@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
 
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
 
    #save the file
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)
 
    return {"filename": file.filename}

@app.get("/show/")
async def read_random_file():
 
    # get random file from the image directory
    files = os.listdir(IMAGEDIR)
    random_index = randint(0, len(files) - 1)
 
    path = f"{IMAGEDIR}{files[random_index]}"
     
    return FileResponse(path)