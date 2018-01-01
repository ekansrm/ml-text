import jieba
import multiprocessing
import codecs
from gensim.models.word2vec import Word2Vec


class WikiData:
    def __init__(self, wiki_file):
        self.wiki_file = wiki_file

    def __iter__(self):
        for line in self.wiki_file:
            s = jieba.cut(line)
            # 使用filter过滤掉空格和换行，可以考虑同时过滤掉英文单词
            s = filter(lambda x: x != ' ' and x != '\n', s)
            yield list(s)


wiki_data = WikiData(codecs.open('zhwiki.txt', 'r', encoding='utf-8'))
model = Word2Vec(wiki_data, size=64, window=5, min_count=5, workers=multiprocessing.cpu_count())

model.save('word2vec.64.model')
