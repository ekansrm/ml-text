from __future__ import unicode_literals, division

from nltk import LidstoneProbDist, compat, ConditionalFreqDist, ngrams
from nltk.book import text6
from nltk.probability import (FreqDist)

from move_this.model import BaseNgramModel


# Natural Language Toolkit: API for Language Models
#
# Copyright (C) 2001-2013 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT
# should this be a subclass of ConditionalProbDistI?
# Natural Language Toolkit: Language Models
#
# Copyright (C) 2001-2014 NLTK Project
# Authors: Steven Bird <stevenbird1@gmail.com>
#          Daniel Blanchard <dblanchard@ets.org>
#          Ilia Kurenkov <ilia.kurenkov@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
# from nltk.util import ngrams
#
# bigrams = ngrams(text6, 2)
# bigramsDist = FreqDist(bigrams)
# print(bigramsDist.most_common(10))


# TODO 这里修改过, 补回 B()
class FixedConditionalFreqDist(ConditionalFreqDist):
    def B(self):
        return len(self)


def _estimator(fdist, *estimator_args, **estimator_kwargs):
    """
    Default estimator function using a SimpleGoodTuringProbDist.
    """
    # can't be an instance method of NgramModel as they
    # can't be pickled either.
    return LidstoneProbDist(fdist, *estimator_args, **estimator_kwargs)


# Ngram模型需要的NgramCounter


class BaseNgramCounter(object):
    def __init__(self, order, super_ngrams, train, pad_left=True, pad_right=False,
                 estimator=None, left_pad_symbol=None, right_pad_symbol=None, *estimator_args, **estimator_kwargs):

        if super_ngrams is not None:
            self.ngrams = super_ngrams
        else:
            self.ngrams = {}

        self.order = order

        # protection from cryptic behavior for calling programs
        # that use the pre-2.0.2 interface
        assert (isinstance(pad_left, bool))
        assert (isinstance(pad_right, bool))

        # make sure n is greater than zero, otherwise print it
        assert (order > 0), order

        # For explicitness save the check whether this is a unigram model
        self.is_unigram_model = (order == 1)
        # save the ngram order number
        self._n = order

        # save left and right padding
        self._pad_left = pad_left
        self._pad_right = pad_right

        if estimator is None:
            estimator = _estimator

        # TODO ConditionalFreqDist 不再从 Freqdict 继承, 没有 freqdict 方法和 B 方法, 所以另外记录 freqDict
        cfd = ConditionalFreqDist()
        freqdict = FreqDist()

        # set read-only ngrams set (see property declaration below to reconfigure)
        self._ngrams = set()

        # If given a list of strings instead of a list of lists, create enclosing list
        if (train is not None) and isinstance(train[0], compat.string_types):
            train = [train]

        for sent in train:
            raw_ngrams = ngrams(sent, order, pad_left, pad_right, left_pad_symbol=' ', right_pad_symbol=' ')
            for ngram in raw_ngrams:
                self._ngrams.add(ngram)
                context = tuple(ngram[:-1])
                token = ngram[-1]
                # TODO 这里修改过
                cfd[context][token] += 1
                freqdict[ngram] += 1

        self.ngrams[order] = freqdict

        # recursively construct the lower-order models
        if not self.is_unigram_model:
            lower_ngrams = BaseNgramCounter(order - 1,
                                            self.ngrams,
                                            train,
                                            pad_left,
                                            pad_right,
                                            )
            self.ngrams[order-1] = lower_ngrams

    def check_against_vocab(self, word):
        if word in self.ngrams[self.order]:
            return word
        return ''

        # raise NotImplementedError()

    def to_ngrams(self, text):
        return ngrams(text, self.order, self._pad_left, self._pad_right, left_pad_symbol=' ', right_pad_symbol=' ')
        # raise NotImplementedError()


class NgramsModel2(BaseNgramModel):

    def __init__(self, n, train, pad_left=True, pad_right=False,
                 estimator=None, *estimator_args, **estimator_kwargs):

        ngramCounter = BaseNgramCounter(n, None, train=text6)

        super(NgramsModel2, self).__init__(ngram_counter=ngramCounter)

    def _words_following(self, context, cond_freq_dist):

        # TODO 需要做兼容处理 Py2 Py3
        # for ctxt, word in cond_freq_dist.items():
        #     if ctxt == context:
        #         yield word.values()
        if context in cond_freq_dist:
            return cond_freq_dist[context]

    def score(self, word, context):
        return self.prob(word=word, context=context)

    def prob(self, word, context):
        """
        Evaluate the probability of this word in this context using Katz Backoff.

        :param word: the word to get the probability of
        :type word: str
        :param context: the context the word is in
        :type context: list(str)
        """
        context = tuple(context)
        _ngrams = context + (word,)
        if _ngrams in self._ngrams or self.is_unigram_model:
            # TODO 这里有修改
            _prob = self._probdist.prob(_ngrams)
            return _prob
        else:
            return self._alpha(context) * self._backoff.prob(word, context[1:])

    def _alpha(self, context):
        """Get the backoff alpha value for the given context
        """
        error_message = "Alphas and backoff are not defined for unigram models"
        assert not self.is_unigram_model, error_message

        if context in self._backoff_alphas:
            return self._backoff_alphas[context]
        else:
            return 1


class NgramsModel(BaseNgramModel):
    def __init__(self, n, train, pad_left=True, pad_right=False,
                 estimator=None, *estimator_args, **estimator_kwargs):

        # 检查参数类型
        assert (isinstance(pad_left, bool))
        assert (isinstance(pad_right, bool))

        # 确保阶数大于零
        assert (n > 0), n

        # 判断是否为unigram
        self.is_unigram_model = (n == 1)
        # 保存模型阶数
        self._n = n

        # 保存padding, padding长度最多为 n-1
        self._lpad = ('',) * (n - 1) if pad_left else ()
        self._rpad = ('',) * (n - 1) if pad_right else ()

        if estimator is None:
            estimator = LidstoneProbDist

        # train期待类型: List<List<String>>
        if (train is not None) and isinstance(train[0], compat.string_types):
            train = [train]

        # TODO ConditionalFreqDist 不再从 Freqdict 继承, 没有 freqdict 方法和 B 方法, 所以另外记录 freqDict

        # cfd 用来计算方法 c(w_i|w_(i-1)w_(i-2)...w(i-(n-1)))
        self.c = ConditionalFreqDist()
        freqdict = FreqDist()

        # ngrams的tuple的集合
        self._ngrams = set()

        for sent in train:
            # ['a', 'b', 'c'] -> [('a', 'b'), ('b', 'c')]
            raw_ngrams = ngrams(sent, n, pad_left, pad_right, left_pad_symbol=' ', right_pad_symbol=' ')
            for ngram in raw_ngrams:
                ngram = tuple(ngram)
                self._ngrams.add(ngram)
                context = ngram[:-1]
                token = ngram[-1]
                self.c[context][token] += 1

                freqdict[ngram] += 1

        # 通过概率估计器得到关于 ngrams 的概率字典
        self._prob_dist = estimator(freqdict, *estimator_args, **estimator_kwargs)

        # 如果不是 unigram, 我们需要生产低阶 ngrams 模型
        if not self.is_unigram_model:
            self._backoff = NgramsModel(n - 1, train,
                                        pad_left, pad_right,
                                        estimator,
                                        *estimator_args,
                                        **estimator_kwargs)

            self._backoff_alphas = dict()

            # 对于每一个条件(w_(i-1), w_(i-2), ..., w(i-(n-1))
            for ctxt in self.c.conditions():
                backoff_ctxt = ctxt[1:]
                backoff_total_pr = 0.0
                total_observed_pr = 0.0

                # this is the subset of words that we OBSERVED following
                # this context.
                # i.e. Count(word | context) > 0
                for word in self._words_following(ctxt):
                    _pr = self.prob(word, ctxt)
                    total_observed_pr += _pr
                    # we also need the total (n-1)-gram probability of
                    # words observed in this n-gram context
                    backoff_total_pr += self._backoff.prob(word, backoff_ctxt)

                assert (0 <= total_observed_pr <= 1), total_observed_pr
                # beta is the remaining probability weight after we factor out
                # the probability of observed words.
                # As a sanity check, both total_observed_pr and backoff_total_pr
                # must be GE 0, since probabilities are never negative
                beta = 1.0 - total_observed_pr

                # backoff total has to be less than one, otherwise we get
                # an error when we try subtracting it from 1 in the denominator
                assert (0 <= backoff_total_pr < 1), backoff_total_pr
                alpha_ctxt = beta / (1.0 - backoff_total_pr)

                self._backoff_alphas[ctxt] = alpha_ctxt

    def _words_following(self, context):

        # TODO 需要做兼容处理 Py2 Py3
        if context in self.c:
            return self.c[context].keys()

    def _alpha(self, context):
        """Get the backoff alpha value for the given context
        """
        assert not self.is_unigram_model, "Alphas and backoff are not defined for unigram models"

        if context in self._backoff_alphas:
            return self._backoff_alphas[context]
        else:
            return 1

    def score(self, word, context):
        return self.prob(word=word, context=context)

    def prob(self, word, context):
        """
        通过 katz Backoff 模型估计输入的词在输入的上下文中的概率

        :param word: the word to get the probability of
        :type word: str
        :param context: the context the word is in
        :type context: list(str)
        """
        context = tuple(context)
        _ngrams = context + (word,)
        if _ngrams in self._ngrams or self.is_unigram_model:
            # TODO 这里有修改
            _prob = self._prob_dist.prob(_ngrams)
            return _prob
        else:
            return self._alpha(context) * self._backoff.prob(word, context[1:])

    def logprob(self, word, context):
        """
        Evaluate the (negative) log probability of this word in this context.

        :param word: the word to get the probability of
        :type word: str
        :param context: the context the word is in
        :type context: list(str)
        """
        return self.logscore(word=word, context=context)


m = NgramsModel(n=2, train=text6, gamma=1, bins=None)

# print(m.prob("think", ("I", "don't")))
# print(m.prob("of", ("The", "Book",)))
print(m.prob("Journal", ("Street",)))
print(m.prob("think", ("I",)))
print(m.prob("don't", ("I",)))
print(m.prob("way", ("Go",)))


m2 = NgramsModel2(n=2, train=text6, gamma=1, bins=None)

print(m2.prob("Journal", ("Street",)))
print(m2.prob("think", ("I",)))
print(m2.prob("don't", ("I",)))
print(m2.prob("way", ("Go",)))

