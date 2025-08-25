# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import karaoke

# Criando a instância do FastAPI
app = FastAPI(title="Audio Metrics API", version="3.4.0")

# Permitir CORS para todas as origens
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite qualquer origem, pode ser restrito a domínios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluindo as rotas definidas no arquivo de rotas
app.include_router(karaoke.router)

