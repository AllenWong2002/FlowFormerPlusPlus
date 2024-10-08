import torch
def build_network(cfg):
    name = cfg.network 
    if name == 'BOFNet':
        from .BOFNet.network import BOFNet as network
    else:
        raise ValueError(f"Network = {name} is not a valid optimizer!")

    return network(cfg[name])
