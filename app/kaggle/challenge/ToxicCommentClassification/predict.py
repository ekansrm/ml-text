from data.kaggle.challenge.ToxicCommentClassification import path_train as TRAIN_DATA_FILE, path_test as TEST_DATA_FILE

test_df = pd.read_csv(TEST_DATA_FILE)

model = load_model(model_checkpoint_best_path)

print('Start making the submission before fine-tuning')

y_test = model.predict([test_data], batch_size=1024, verbose=1)

sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission[list_classes] = y_test

sample_submission.to_csv('%.4f_' % bst_val_score + '.csv', index=False)

from . import DATA_TRAIN, DATA_TEST, DATA


