from app.api.routers import machine_learn_controller
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title = 'Flight Machine Learn',
    version = '1.0.0',
    description = 'Uma Análise e predição de dados de Machine Learn'
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(machine_learn_controller.router)
app.mount("/plots", StaticFiles(directory="../Data/Plots"), name="plots")