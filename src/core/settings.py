# --------------------------------------------------------------------------
# Transformer 및 여러 Configuration, Settings 값을 설정하는 모듈입니다.
# --------------------------------------------------------------------------


class TransformerSettings:
    """
    Transformer 설정
    """
    embedding_dim: int = 768
    embedding_dropout: float = 0.1
    vocab_size: int = 30000
    max_len: int = 512
    trainable_pos_embedding: bool = True
    num_heads: int = 12
    num_layers: int = 12
    attention_dropout: float = 0.1
    use_one_embedding_dropout: bool = False
    d_hid: int = 768 * 4
    residual_dropout: float = 0.1
    use_attn_mask: bool = True
    embedding_layer_norm: bool = False
    neg_inf: float = -1e9
    layer_norm_epsilon: float = 1e-5
    accurate_gelu: bool = False

    def __init__(self):
        pass
