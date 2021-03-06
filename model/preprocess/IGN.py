# 导入使用到的库
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as sk_preprocessing
import pandas as pd


df = pd.read_csv('../data/ign/ign.csv').iloc[:, 1:3]
df.score_phrase.value_counts()
df = df[df.score_phrase != 'Disaster']


title = df['title']
label = df['score_phrase']


# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(title, label, test_size=0.1, random_state=42)

# 对类别变量进行编码，共10类
y_labels = list(y_train.value_counts().index)
le = sk_preprocessing.LabelEncoder()
le.fit(y_labels)
num_labels = len(y_labels)
y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)
y_test = to_categorical(y_test.map(lambda x: le.transform([x])[0]), num_labels)

# 分词，构建单词-id词典
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
tokenizer.fit_on_texts(title)
vocab = tokenizer.word_index

# 将每个词用词典中的数值代替
X_train_word_ids = tokenizer.texts_to_sequences(X_train)
X_test_word_ids = tokenizer.texts_to_sequences(X_test)
