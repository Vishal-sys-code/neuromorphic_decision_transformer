def get_model(cfg):
    model_name = cfg.model.name
    if model_name == "dt":
        from src.models.dt import DecisionTransformer
        return DecisionTransformer(cfg)
    elif model_name == "snn_dt":
        from src.models.snn_dt import SnnDt
        return SnnDt(cfg)
    elif model_name == "iql":
        from src.models.iql import IQL
        return IQL(cfg)
    elif model_name == "cql":
        from src.models.cql import CQL
        return CQL(cfg)
    elif model_name == "dsformer":
        from src.models.dsformer import DsFormer
        return DsFormer(cfg)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")