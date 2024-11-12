from transformers import BertTokenizer
import html2text
from bs4 import BeautifulSoup
import torch


class Preprocessor:
    def __init__(self, max_token_len=512):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_token_len = max_token_len
        self.html_parser = html2text.HTML2Text()
        self.html_parser.ignore_links = True

    def preprocess_url(self, url: str):
        # URL 토큰화
        tokens = self.tokenizer(
            url,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokens["input_ids"], tokens["attention_mask"]

    def preprocess_html(self, html_content: str):
        # HTML 텍스트 추출 및 토큰화
        soup = BeautifulSoup(html_content, "html.parser")
        text = self.html_parser.handle(str(soup))
        tokens = self.tokenizer(
            text,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokens["input_ids"], tokens["attention_mask"]
