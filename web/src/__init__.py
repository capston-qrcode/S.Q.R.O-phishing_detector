import logging

from fastapi import FastAPI
from contextlib import asynccontextmanager

from web.src.core.settings import AppSettings

version = "0.1.0"

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Application startup")
        yield
    finally:
        logger.info("Application shutdown")


def create_app(app_settings: AppSettings) -> FastAPI:
    app = FastAPI(
        version=version,
    )

    # if settings.BACKEND_CORS_ORIGINS:
    #     app.add_middleware(
    #         CORSMiddleware,
    #         allow_origins=[
    #             str(origin).strip("/") for origin in settings.BACKEND_CORS_ORIGINS
    #         ],
    #         allow_credentials=True,
    #         allow_methods=["*"],
    #         allow_headers=["*"],
    #     )

    return app
