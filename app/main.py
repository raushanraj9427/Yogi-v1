from fastapi import FastAPI, HTTPException, Depends
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

from .models import Plants , Details
from sqlalchemy.orm import Session
from .db import  get_db
from fastapi_sqlalchemy import DBSessionMiddleware
import os
load_dotenv(".env")
app = FastAPI()
# app.add_middleware(DBSessionMiddleware, db_url = os.environment["DATABASE_URL"])

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



@app.get("/plants/{plant_id}")
async def read_plant(plant_id: int, db: Session = Depends(get_db)):
    if (
        result := db.query(Plants)
        .filter(Plants.id == plant_id)
        .first()
    ):
        return result
    else:
        raise HTTPException(status_code=404, detail='Questions is not found')

@app.get("/details/(plant_id)")
async def read_detail(plant_id:int, db: Session = Depends(get_db)):
    if (
        result := db.query(Details)
        .filter(Details.plant_id == plant_id)
        .first()
    ):
        return result
    else:
        raise HTTPException(status_code=404, detail='Questions is not found')

@app.post("/plants/")
async def create_plant(plant_text:str, db: db_dependency):
    db_plant = models.Plants(plant_text = plant_text)
    db.add(db_plant)
    db.commit()
    db.refresh(db_plant)


@app.post("/plants_details/")
async def create_plant_details(plant: PlantBase, db: db_dependency):
    db_plant = models.Plants(plant_text = plant.plant_text)
    db.add(db_plant)
    db.commit()
    db.refresh(db_plant)
    for choice in plant.choices:
        db_choice = models.Details(plant_family=choice.plant_family, plant_bio=choice.plant_bio, plant_descr=choice.plant_descr, plant_url= choice.plant_url, plant_id=db_plant.id)
        db.add(db_choice)
    db.commit()