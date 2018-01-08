from nltk import FreqDist
from nltk import ngrams
from nltk.book import text6

bigrams = ngrams(text6, 2)
bigramsDist = FreqDist(bigrams)
print(bigramsDist.most_common(10))
