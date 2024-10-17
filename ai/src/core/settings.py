# --------------------------------------------------------------------------
# Transformer 및 여러 Configuration, Settings 값을 설정하는 모듈입니다.
# --------------------------------------------------------------------------


class TransformerSettings:
    """
    Transformer 설정
    """

    embedding_dim: int = 768
    max_token_len: int = 512
    embedding_dropout: float = 0.1
    vocab_size: int = 30000
    trainable_pos_embedding: bool = True
    num_heads: int = 12
    num_layers: int = 12
    use_one_embedding_dropout: bool = False
    use_attn_mask: bool = True
    embedding_layer_norm: bool = False
    neg_inf: float = -1e9
    accurate_gelu: bool = False

    # 논문 참고 값
    learning_rate: float = 1e-5
    batch_size: int = 8
    epoch: int = 100
    optimizer: str = "Adam"
    residual_dropout: float = 0.1
    attention_dropout: float = 0.1

    def __init__(self):
        pass
