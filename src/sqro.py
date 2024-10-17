# 예제 TEXT 분석 모듈 (from ChatGPT), 예시 소스코드이므로 가까운 커밋 내에 제거 예정.
from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs
from keras.api.optimizers import Adam
from keras.api.layers import Dense
from keras.api import Model

import numpy as np


# BERT 모델의 토크나이저에 필요한 사전 생성
def create_token_dict():
    token_dict = get_base_dict()  # BERT의 기본 토큰 생성
    return token_dict


# 샘플 데이터 생성 (X: 문장, y: 레이블)
X = [
    "Keras-bert is great for text classification",
    "I love machine learning",
    "BERT is powerful",
]
y = [1, 1, 0]  # 간단한 이진 분류 문제 (1: 긍정, 0: 부정)

# 사전 생성
token_dict = create_token_dict()


# 배치 데이터 생성기
def create_batch_inputs(X, y, token_dict, seq_len=128):
    indices, segments = [], []
    for sentence in X:
        tokens = sentence.split()
        indices.append(
            [token_dict.get(token, token_dict["[UNK]"]) for token in tokens]
        )
        segments.append([0] * len(indices[-1]))
    indices = np.array(
        [np.pad(i, (0, seq_len - len(i)), "constant") for i in indices]
    )
    segments = np.array(
        [np.pad(s, (0, seq_len - len(s)), "constant") for s in segments]
    )
    y = np.array(y)
    return indices, segments, y


# BERT 모델 정의
def build_bert_model():
    model = get_model(
        token_num=len(token_dict),
        head_num=12,
        transformer_num=12,
        embed_dim=768,
        feed_forward_dim=3072,
        seq_len=128,
    )

    # 분류를 위한 Dense 레이어 추가
    output = Dense(1, activation="sigmoid")(model.output)

    # 최종 모델 구성
    model = Model(model.input, output)

    # 모델 컴파일
    compile_model(model)
    return model


# 모델 학습
def train_model(X, y):
    # 입력 데이터 준비
    indices, segments, labels = create_batch_inputs(X, y, token_dict)

    # 모델 생성
    model = build_bert_model()

    # 모델 컴파일
    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # 모델 학습
    model.fit([indices, segments], labels, epochs=3, batch_size=2)

    return model


# 학습 실행
model = train_model(X, y)
