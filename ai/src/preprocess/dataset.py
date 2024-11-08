# --------------------------------------------------------------------------
# 파서의 데이터를 읽고 토큰화 시키는 모듈입니다.
# --------------------------------------------------------------------------
import sqlite3
import html2text

import pandas as pd

from bs4 import BeautifulSoup

from ai.src.utils.logging import setup_logging
from ai.src.preprocess.language import filter_english


class DataConnector:
    def __init__(self, settings):
        self.logger = setup_logging(settings=settings)
        self.db_path = None  # TODO : 설정에 따라 DB 경로 지정

    def _fetch_db(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM table_name", conn)  # TODO : 테이블명 변경
        conn.close()
        return df

    def get_filtered_data(self):
        """
        DB에서 불러온 데이터를 필터링하여 영어 텍스트만 반환합니다.
        """
        df = self._fetch_db()
        df["filtered_content"] = df["html_content"].apply(filter_english)
        df = df.dropna(subset=["filtered_content"])
        return df


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
        return self.parser.handle(html_content)

    def preprocess_text(self, html_content):
        urls = self._extract_urls(html_content)
        text = self._extract_html_text(html_content)

        filtered_text = filter_english(text)
        return urls, filtered_text if filtered_text else ""
