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

@app.post("/get_answer")
async def get_answer(request:Request,question:str=Form(...)):
    print(question)
    result = get_result(query=question)
    response_data = jsonable_encoder(json.dumps({"answer":result}))
    res = Response(response_data)
    print(res)
    return res


if __name__ == "__main__":
    uvicorn.run("app:app",host="0.0.0.0",port=8000,reload=True)