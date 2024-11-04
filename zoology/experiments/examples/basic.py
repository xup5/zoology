from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig
from zoology.data.associative_recall import MQARConfig


VOCAB_SIZE = 8_192

config = TrainConfig(
    data=DataConfig(
        train_configs=[MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000, num_kv_pairs=8)],
        test_configs=[MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=8)],
        cache_dir="/n/home13/xupan/Projects/token_compression/zoology_cache",
        vocab_size=VOCAB_SIZE,
        input_seq_len=128,
        num_train_examples=20_000,
        num_test_examples=1_000,
        builder=FunctionConfig(
            name="zoology.data.associative_recall.multiquery_ar",
            kwargs={"num_kv_pairs": 8}
        ),
        
    ),
    model=ModelConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=128,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.attention.MHA",
            kwargs={"dropout": 0.1, "num_heads": 1}
        )
    ),
    
)

configs = [config]