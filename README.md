# S.Q.R.O-PhishingDetector

BERT 모델 기반 멀티모달 피싱 사이트 탐지기, Secure QR Operations(S.Q.R.O)의 메인 소스코드 저장소입니다.

## Stacks

- Lang / Model : Python 3.12.4 / BERT
- English content filtering : langdetect
- HTML content -> Text 변환 : HTML2Text
- HTML raw -> URL 추출 : Beautiful Soup HTML parser
- BERT의 keras 구현체 : keras-bert
- formatter / linter : black&isort / flake8

# Guide

작업 후 저장소에 코드를 업로드 하기 전, 하단의 스크립트들을 차례로 실행해주세요.

## 1. Testing

> test 스크립트 및 testcase는 핵심 로직 구현과 동시에 작업될 예정입니다.

```bash
bash scripts/run-test.sh
```

## 2. commit 전, 스크립트 실행

작업 후 커밋 전 하단의 스크립트들을 **_반드시 실행하여_** 코드 포멧팅 및 구문 검사를 수행해주세요.

### Formatting (Black)

```bash
bash scripts/format.sh
```

### Linting (Flake8)

```bash
bash scripts/lint.sh
```
