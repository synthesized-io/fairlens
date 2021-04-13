from dataclasses import dataclass


@dataclass
class ModelBuilderConfig(PersonModelConfig):
    categorical_threshold_log_multiplier: float = 2.5
    min_num_unique: int = 10

