from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Annotated
import models
from database import engine, SessionLocal
from sqlalchemy.orm import Session


app = FastAPI()
models.Base.metadata.create_all(bind=engine)

class DetailBase(BaseModel):
    plant_family: str
    plant_bio: str
    plant_descr: str
    plant_url: str

class PlantBase(BaseModel):
    plant_text: str
    choices: List[DetailBase]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()   

db_dependency = Annotated[Session, Depends(get_db)]

@app.get("/plants/{plant_id}")
async def read_plant(plant_id: int, db: db_dependency):
    result = db.query(models.Plants).filter(models.Plants.id == plant_id).first()
    if not result:
        raise HTTPException(status_code=404, detail='Questions is not found')
    return result

@app.get("/details/(plant_id)")
async def read_detail(plant_id:int, db: db_dependency):
    result = db.query(models.Details).filter(models.Details.plant_id == plant_id).first()
    if not result:
        raise HTTPException(status_code=404, detail='Questions is not found')
    return result

@app.post("/plants/")
async def create_plants(plant: PlantBase, db: db_dependency):
    db_plant = models.Plants(plant_text = plant.plant_text)
    db.add(db_plant)
    db.commit()
    db.refresh(db_plant)
    for choice in plant.choices:
        db_choice = models.Details(plant_family=choice.plant_family, plant_bio=choice.plant_bio, plant_descr=choice.plant_descr, plant_url= choice.plant_url, plant_id=db_plant.id)
        db.add(db_choice)
    db.commit()