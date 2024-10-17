import pytest
import pytest_asyncio

from fastapi import status
from httpx import AsyncClient

from web.tests.conftest import BaseTestRouter


@pytest.mark.asyncio
class TestSQROAPI(BaseTestRouter):
    @pytest_asyncio.fixture(autouse=True)
    async def setup_method(self, client: AsyncClient):
        pass

    @pytest_asyncio.fixture(autouse=True)
    async def teardown_method(self, client: AsyncClient):
        pass

    async def test_get_api(self, client: AsyncClient):
        # given

        # when
        response = await client.get("/some/api/endpoint")

        # then
        ok = status.HTTP_200_OK
        print(response)
