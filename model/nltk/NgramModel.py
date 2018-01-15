"""
NLTK 的 Ngram 模型. 在 2.x.x 版本原本是存在的, 不过在 3.0.0 版本有, 由于很多类和函数变化太大, nltk.model 无法迁移, 被移出分支.
至今仍未合并. 这个代码是bugfix后的代码

"""

from math import log
from nltk.probability import (FreqDist)
from nltk import LidstoneProbDist, compat, ConditionalFreqDist, ngrams

NEG_INF = float("-inf")


def _estimator(fdist, *estimator_args, **estimator_kwargs):
    """
    Default estimator function using a SimpleGoodTuringProbDist.
    """
    # can't be an instance method of NgramModel as they
    # can't be pickled either.
    return LidstoneProbDist(fdist, *estimator_args, **estimator_kwargs)


class NgramsModel(object):
    def __init__(self, n, train, pad_left=True, pad_right=False,
                 estimator=None, *estimator_args, **estimator_kwargs):

        # protection from cryptic behavior for calling programs
        # that use the pre-2.0.2 interface
        assert (isinstance(pad_left, bool))
        assert (isinstance(pad_right, bool))

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

        cfd = ConditionalFreqDist()
        freq_dict = FreqDist()

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
                freq_dict[ngram] += 1

        self._prob_dist = estimator(freq_dict, *estimator_args, **estimator_kwargs)

        # recursively construct the lower-order models
        if not self.is_unigram_model:
            self._backoff = NgramsModel(n - 1, train,
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
                    _pr = self.score(word, ctxt)
                    total_observed_pr += _pr
                    # we also need the total (n-1)-gram probability of
                    # words observed in this n-gram context
                    backoff_total_pr += self._backoff.score(word, backoff_ctxt)

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

    @staticmethod
    def _words_following(context, cond_freq_dist):

        # TODO cond_freq_dist 的数据类型已经发生了变化. 另外, 下面的方式效率更高些
        # for ctxt, word in cond_freq_dist.items():
        #     if ctxt == context:
        #         yield word.values()
        if context in cond_freq_dist:
            return cond_freq_dist[context]

    def score(self, word, context):
        """
        Evaluate the probability of this word in this context using Katz Backoff.

        :param word: the word to get the probability of
        :type word: str
        :param context: the context the word is in
        :type context: list(str)
        """

        # 这里要考虑 order=1 的情况
        if self.is_unigram_model:
            _ngrams = (word,)
        else:
            context = tuple(context)
            _ngrams = context + (word,)

        if _ngrams in self._ngrams or self.is_unigram_model:
            # TODO 这里有修改
            _prob = self._prob_dist.prob(_ngrams)
            return _prob
        else:
            return self._alpha(context) * self._backoff.score(word, context[1:])

    def _alpha(self, context):
        """Get the backoff alpha value for the given context
        """
        error_message = "Alphas and backoff are not defined for unigram models"
        assert not self.is_unigram_model, error_message

        if context in self._backoff_alphas:
            return self._backoff_alphas[context]
        else:
            return 1

    def logscore(self, word, context):
        """
        Evaluate the (negative) log probability of this word in this context.

        :param word: the word to get the probability of
        :type word: str
        :param context: the context the word is in
        :type context: list(str)
        """
        score = self.score(word, context)
        if score == 0.0:
            return NEG_INF
        return log(score, 2)


if __name__ == '__main__':

    from nltk.book import text6
    m1 = NgramsModel(n=1, train=text6, gamma=1, bins=None)

    print('-'*8 + "n=01" + '-'*8)
    print(m1.score("Journal", ()))
    print(m1.score("think", ()))
    print(m1.score("don't", ()))
    print(m1.score("way", ()))
    print()

    print('-'*8 + "n=02" + '-'*8)
    m2 = NgramsModel(n=2, train=text6, gamma=1, bins=None)
    print(m2.score("Journal", ("Street",)))
    print(m2.score("think", ("I",)))
    print(m2.score("don't", ("I",)))
    print(m2.score("way", ("Go",)))
    print()

    print('-'*8 + "n=03" + '-'*8)
    m3 = NgramsModel(n=3, train=text6, gamma=1, bins=None)
    print(m2.score("Journal", ("Wall", "Street",)))
    print(m2.score("think", ("I", "don't", )))
    print(m2.score("don't", ("I", "really")))
    print(m2.score("way", ("Just", "go",)))
    print()
