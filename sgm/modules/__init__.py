from .encoders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "sgm.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
