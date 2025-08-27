# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import karaoke

# ===================== API =====================
app = FastAPI(title="Audio Metrics API (sounddevice)", version="3.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# Incluindo as rotas definidas no arquivo de rotas
app.include_router(karaoke.router)

˜