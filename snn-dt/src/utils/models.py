from src.models.cql import CQL
from src.models.dt import DecisionTransformer
from src.models.dsformer import DsFormer
from src.models.iql import IQL
from src.models.snn_dt import SnnDt


def get_model(cfg):
    if cfg.model.name == "dt":
        return DecisionTransformer(cfg)
    elif cfg.model.name == "snn_dt":
        return SnnDt(cfg)
    elif cfg.model.name == "iql":
        return IQL(cfg)
    elif cfg.model.name == "cql":
        return CQL(cfg)
    elif cfg.model.name == "dsformer":
        return DsFormer(cfg)
    else:
        raise NotImplementedError(f"Model {cfg.model.name} not implemented.")