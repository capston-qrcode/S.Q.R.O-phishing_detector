# --------------------------------------------------------------------------
# 파서의 데이터를 읽고 토큰화 시키는 모듈입니다.
# --------------------------------------------------------------------------
import html2text

from bs4 import BeautifulSoup
from typing import NamedTuple


class TokenizedData(NamedTuple):
    pass


def read_db() -> None:
    """
    sqlite3 DB(파서로부터 생성된 .db 확장자 파일) 읽어오는 메서드

    :return:
    """
    pass


def extract_urls(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    urls = []
    for tag in soup.find_all(['a', 'link']):
        url = tag.get('href')
        if url:
            urls.append(url)
    return urls


def extract_html_text(html_content):
    parser = html2text.HTML2Text()
    parser.ignore_links = True
    text = parser.handle(html_content)
    return text
