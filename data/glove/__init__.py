from setup import config as base_config

config = base_config.cast("glove")


def path(p: str)->str:
    return config.data.path(p)


path_glove_840B_300d = path("glove.840B.300d.txt")
