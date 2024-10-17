from __future__ import annotations

import uvicorn

from web.src import create_app
from web.src.core.settings import settings


app = create_app(settings)

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=settings.SERVER_DOMAIN,
        port=settings.SERVER_PORT,
        reload=True,
    )
