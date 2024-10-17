import pytest_asyncio

from httpx import AsyncClient, ASGITransport

from web.src import create_app
from web.src.core.settings import AppSettings


app_settings = AppSettings(_env_file=".env.test")


class BaseTestRouter:
    @pytest_asyncio.fixture(scope="function")
    async def client(self):
        app = create_app(app_settings)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c
