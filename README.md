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

개발 환경은 Python 3.12.4 버전을 기준으로 세팅되어 있습니다.   
따라서 로컬에 Python 3.12.4 버전이 필수로 설치되어 있어야 합니다.

pyenv를 활용하면 쉽게 로컬 파이썬 버전을 수정할 수 있습니다 : [가이드](https://blog.flynnpark.dev/7)

협업에 관련된 스크립트들을 정의해두었으니, 하단의 스크립트들을 차례로 정독해주세요.

## 0. 가상 환경 설치

S.Q.R.O 프로젝트의 dependency를 다른 프로젝트와 분리하기 위해 다음 스크립트를 실행하여 가상환경을 셋업합니다.

```bash
# 가상 환경 세팅
python -m venv venv

# 가상 환경 진입 (MacOS & Linux)
source venv/bin/activate

# 가상 환경 진입 (Windows)
.\venv\Scripts\activate
```

터미널 커맨드라인 앞에 (venv)가 보이면 가상 환경 진입에 성공한 것입니다.

가상 환경 진입 후, pip 혹은 poetry를 활용하여 개발 및 구동에 필요한 dependency 들을 설치합니다.

```bash
# pip 사용하는 경우 (용량이 크기 때문에 반드시 가상환경 진입 후 실행할 것)
pip install -r requirements.txt

# poetry 사용하는 경우 (용량이 크기 때문에 반드시 가상환경 진입 후 실행할 것)
poetry install
```

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
