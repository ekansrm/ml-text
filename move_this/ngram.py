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
from nltk.util import ngrams

from move_this.ngramLegacy import NgramModel

bigrams = ngrams(text6, 2)
bigramsDist = FreqDist(bigrams)
print(bigramsDist.most_common(10))

from nltk.corpus import brown
from nltk.probability import LidstoneProbDist
est = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
lm = NgramModel(3, brown.words(categories='news'), estimator=est)

print(lm.prob("stupid", "you are"))

print(lm.prob("cup", "you are"))
