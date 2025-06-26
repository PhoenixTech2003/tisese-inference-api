from fastapi import FastAPI
from .routers import inference
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()
origins = [
    "http://localhost:300"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["POST"],
    allow_headers=["*"]
)

app.include_router(inference.router)

