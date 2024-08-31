from fastapi import FastAPI, Request, Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import uvicorn
import json
import os
from dotenv import load_dotenv
from src.retrival import get_result

load_dotenv()


app = FastAPI()

templates = Jinja2Templates("templates")

@app.get("/")
async def index(request:Request):
    return  templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_answers")
async def get_answers(request:Request,question:str=Form(...)):
    pass