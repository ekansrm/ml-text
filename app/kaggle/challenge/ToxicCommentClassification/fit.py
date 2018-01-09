import os
import pandas as pd

from keras.preprocessing.text import Tokenizer
from . import DAT_TRAIN, DAT_TEST, DAT


# 获取TOKENIZER
# 获取 EMBEDDING 向量 (可选)

if __name__ == '__main__':

    if os.path.exists(DAT('train.sample.csv')):
        train_df = pd.read_csv(DAT('train.csv'))
        train_sample_df = train_df.sample(n=10000)
        train_sample_df.to_csv(path=DAT('train.sample.csv'), index=False)
    else:
        train_sample_df = pd.read_csv(DAT('train.sample.csv'))

    tokenizer = Tokenizer(num_words=100000)
    tokenizer.fit_on_texts(train_sample_df)






# early_stopping = earlystopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=5, verbose=1)
#
# checkpoint_better = modelcheckpoint(
#     model_checkpoint_better_path, save_best_only=true, monitor='val_acc',  mode='max', verbose=1)
#
# checkpoint_best = modelcheckpoint(
#     model_checkpoint_best_path, save_best_only=true, monitor='val_acc',  mode='max', verbose=1)
