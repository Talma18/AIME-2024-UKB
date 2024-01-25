from transformers import RobertaConfig


class BioBertaConfig(RobertaConfig):
    model_type = "bioberta"

    def __init__(self,  **kwargs):
        self.max_year = 200
        self.pool_method = "mean"
        self.pool_self = False
        self.use_temporal = True
        self.use_encoder = True

        super().__init__(**kwargs)
