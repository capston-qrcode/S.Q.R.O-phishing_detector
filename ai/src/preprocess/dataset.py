# --------------------------------------------------------------------------
# 파서의 데이터를 읽고 토큰화 시키는 모듈입니다.
# --------------------------------------------------------------------------
import sqlite3
from typing import List

import keras
import html2text
import tensorflow as tf

import pandas as pd

from bs4 import BeautifulSoup
from transformers import BertTokenizer

from ai.src.utils.logging import setup_logging
from ai.src.preprocess.language import filter_english


class DataConnector:
    def __init__(self, settings):
        self.logger = setup_logging(settings=settings)
        self.db_path = settings.DB_PATH

    def _fetch_db(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM phishing_data", conn)
        conn.close()
        return df

    def get_filtered_data(self):
        """
        DB에서 불러온 데이터를 필터링하여 영어 텍스트만 반환합니다.
        """
        df = self._fetch_db()
        df["filtered_content"] = df["html_content"].apply(filter_english)
        df = df.dropna(subset=["filtered_content"])
        df['label'] = df['label'].apply(lambda x: 1 if x == 'malicious' else 0)
        return df


class PreProcessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.html_parser = BeautifulSoup
        self.text_parser = html2text.HTML2Text()
        self.text_parser.ignore_links = True
        self.ascii_chars = {chr(i): i for i in range(32, 127)}  # ASCII 32~126
        self.special_tokens = {'[CLS]': 96, '[SEP]': 97, 'Padding': 98, 'None': 99}

    def _extract_urls(self, html_content: str) -> List[str]:
        soup = BeautifulSoup(html_content, "html.parser")
        urls = [tag.get("href") for tag in soup.find_all(["a", "link"]) if tag.get("href")]

        encoded_urls = [self.special_tokens['[CLS]']]
        for url in urls:
            for char in url:
                encoded_urls.append(ord(char) if 32 <= ord(char) < 127 else self.special_tokens['None'])
            encoded_urls.append(self.special_tokens['[SEP]'])

        return encoded_urls

    def _ascii_encode(self, text: str) -> List[int]:
        # ASCII-based encoding with special tokens
        encoded = [self.special_tokens['[CLS]']]
        for char in text:
            encoded.append(self.ascii_chars.get(char, self.special_tokens['None']))
        encoded.append(self.special_tokens['[SEP]'])
        return encoded

    def preprocess_url(self, url: str) -> (tf.Tensor, tf.Tensor):
        url_encoded = self._extract_urls(url)
        padded_url = keras.preprocessing.sequence.pad_sequences(
            [url_encoded], maxlen=512, padding='post', value=self.special_tokens['Padding']
        )
        attention_mask = [1 if i != self.special_tokens['Padding'] else 0 for i in padded_url[0]]
        return tf.convert_to_tensor(padded_url[0]), tf.convert_to_tensor(attention_mask)

    def preprocess_text(self, html_content: str) -> (tf.Tensor, tf.Tensor):
        english_text = filter_english(html_content)
        if not english_text:
            return tf.constant([]), tf.constant([])

        tokens = self.tokenizer(
            english_text, max_length=512, padding="max_length", truncation=True, return_tensors="tf"
        )
        return tokens["input_ids"][0], tokens["attention_mask"][0]
