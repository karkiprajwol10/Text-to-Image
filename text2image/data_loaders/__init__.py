from .CUB_200_2011 import CUB200Dataset




__loadersdict__ = {'CUB': CUB200Dataset,}


def get_loader(name, **kwargs):
    return __loadersdict__[name](**kwargs)


def get_loader_names():
    return __loadersdict__.keys()
