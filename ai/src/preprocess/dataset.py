# --------------------------------------------------------------------------
# 파서의 데이터를 읽고 토큰화 시키는 모듈입니다.
# --------------------------------------------------------------------------
import sqlite3
import html2text

import pandas as pd

from bs4 import BeautifulSoup
from typing import NamedTuple

from ai.src.utils.logging import setup_logging
from ai.src.preprocess.language import filter_english


class DataConnector:
    def __init__(self, settings):
        self.logger = setup_logging(settings=settings)
        self.db_path = None  # TODO : add me

    def _fetch_db(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM table_name", conn
        )  # TODO : change table name

        print(df.head())

    def _filter_data(self):
        # filter_english()
        pass

    def get_filtered_data(self):
        return


class PreProcessor:
    def __init__(self, settings):
        self.logger = setup_logging(settings=settings)
        self.soup = BeautifulSoup()
        self.parser = html2text.HTML2Text()
        self.urls = []

    def _extract_urls(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup.find_all(["a", "link"]):
            url = tag.get("href")
            if url:
                self.urls.append(url)

        return self.urls

    def _extract_html_text(self, html_content):
        self.parser.ignore_links = True
        text = self.parser.handle(html_content)

        return text

    def preprocess_text(self):
        pass

    def get_input_datasets(self):
        pass
