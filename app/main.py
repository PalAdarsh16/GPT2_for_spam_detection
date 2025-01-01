from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, validator
from app import tasks

app = FastAPI()

class spamchecker(BaseModel):
    text : str

@app.get("/")
def get_root():
    return{"message":"Hello World"}

@app.post("/spamcheck")    
def post_check(t: spamchecker, background_tasks: BackgroundTasks):
    
    t_id = tasks.store_checkreq(t)
    background_tasks.add_task(tasks.run_check,t_id)
    return{"tasks_id": t_id}

@app.get("/results")
def get_reply(t_id: int):
    return{"Your input is :": tasks.find(t_id)}
