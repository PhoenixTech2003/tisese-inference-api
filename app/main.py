from fastapi import FastAPI
from .routers import inference
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.include_router(inference.router)

