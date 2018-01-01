from setup import config as base_config

config = base_config.cast("kaggle-toxic-comment-classification-challenge")


def path(p: str)->str:
    return config.data.path(p)


path_train = path("train.csv")
path_test = path("test.csv")
