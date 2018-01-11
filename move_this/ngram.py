from __future__ import unicode_literals, division
from nltk.book import text6

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

from nltk.probability import (FreqDist)
# from nltk.util import ngrams
#
# bigrams = ngrams(text6, 2)
# bigramsDist = FreqDist(bigrams)
# print(bigramsDist.most_common(10))

from move_this.model import BaseNgramModel
from nltk import LidstoneProbDist, compat, ConditionalFreqDist, ngrams


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


class NgramsModel(BaseNgramModel):

    def __init__(self, n, train, pad_left=True, pad_right=False,
                 estimator=None, *estimator_args, **estimator_kwargs):

        # protection from cryptic behavior for calling programs
        # that use the pre-2.0.2 interface
        assert(isinstance(pad_left, bool))
        assert(isinstance(pad_right, bool))

        # make sure n is greater than zero, otherwise print it
        assert (n > 0), n

        # For explicitness save the check whether this is a unigram model
        self.is_unigram_model = (n == 1)
        # save the ngram order number
        self._n = n
        # save left and right padding
        self._lpad = ('',) * (n - 1) if pad_left else ()
        self._rpad = ('',) * (n - 1) if pad_right else ()

        if estimator is None:
            estimator = _estimator

        cfd = FixedConditionalFreqDist()

        # set read-only ngrams set (see property declaration below to reconfigure)
        self._ngrams = set()

        # If given a list of strings instead of a list of lists, create enclosing list
        if (train is not None) and isinstance(train[0], compat.string_types):
            train = [train]

        for sent in train:
            raw_ngrams = ngrams(sent, n, pad_left, pad_right, left_pad_symbol=' ', right_pad_symbol=' ')
            for ngram in raw_ngrams:
                self._ngrams.add(ngram)
                context = tuple(ngram[:-1])
                token = ngram[-1]
                # TODO 这里修改过
                cfd[context][token] += 1

        self._probdist = estimator(cfd, *estimator_args, **estimator_kwargs)

        # recursively construct the lower-order models
        if not self.is_unigram_model:
            self._backoff = NgramsModel(n-1, train,
                                        pad_left, pad_right,
                                        estimator,
                                        *estimator_args,
                                        **estimator_kwargs)

            self._backoff_alphas = dict()
            # For each condition (or context)
            for ctxt in cfd.conditions():
                backoff_ctxt = ctxt[1:]
                backoff_total_pr = 0.0
                total_observed_pr = 0.0

                # this is the subset of words that we OBSERVED following
                # this context.
                # i.e. Count(word | context) > 0
                for word in self._words_following(ctxt, cfd):
                    _pr = self.prob(word, ctxt)
                    print(_pr)
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

    def _words_following(self, context, cond_freq_dist):

        # TODO 需要做兼容处理 Py2 Py3
        # for ctxt, word in cond_freq_dist.items():
        #     if ctxt == context:
        #         yield word.values()
        if context in cond_freq_dist:
            return iter(cond_freq_dist[context])

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
        if context + (word,) in self._ngrams or self.is_unigram_model:
            print(word)
            # TODO 这里有修改
            _prob = self._probdist.prob((word, ))
            print(_prob)
            print(type(_prob))
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

    def logprob(self, word, context):
        """
        Evaluate the (negative) log probability of this word in this context.

        :param word: the word to get the probability of
        :type word: str
        :param context: the context the word is in
        :type context: list(str)
        """
        return self.logscore(word=word, context=context)


module = NgramsModel(n=2, train=text6, gamma=0.2, bins=None)

print(module.prob("Apple", "My name is"))
