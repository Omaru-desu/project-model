import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.api.routes.health import router as health_router
from app.api.routes.segment import router as segment_router
from app.api.routes.search import router as search_router
from app.api.routes.embed import router as embed_router
from app.api.routes.process import router as process_router
from app.services.sam_runtime import sam_runtime
from app.services.dinov3_runtime import dinov3_runtime

load_dotenv()

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
allow_origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]

app = FastAPI(title="fathom")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event() -> None:
    sam_runtime.load()
    dinov3_runtime.load()

app.include_router(health_router, prefix="")
app.include_router(segment_router, prefix="/segment")
app.include_router(search_router, prefix="/search")
app.include_router(embed_router, prefix="/embed")
app.include_router(process_router, prefix="/process")
