from data.kaggle.challenge.ToxicCommentClassification import DAT, DAT_TEST, DAT_TRAIN

from setup import config as base_config

config = base_config.cast("kaggle-toxic-comment-classification-challenge")

LOG = config.log.path

VAR = config.parameter

