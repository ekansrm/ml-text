from setup import config as base_config

config = base_config.cast("kaggle-toxic-comment-classification-challenge")

DAT = config.data.path
DAT_TRAIN = DAT('train.csv')
DAT_TEST = DAT('test.csv')
