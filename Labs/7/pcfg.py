from __future__ import division
import sys
from pprint import pprint
from collections import defaultdict
import nltk
from nltk.corpus import treebank
from nltk import ConditionalFreqDist, Nonterminal, FreqDist
from pcfg_fix import *
from BetterICP import *
from nltk import InsideChartParser

## Main body of code ##
# Extracting tagged sentences using NLTK libraries
psents = treebank.parsed_sents()
# Comment out the following 3 lines if you get tired of seeing them
print "\n 1st parsed sentence: \n", psents[0]
print "\n Productions in the 1st parsed sentence: \n"
pprint(psents[0].productions())

grammar = parse_pgrammar("""
    # Grammatical productions.
     S -> NP VP [1.0]
     NP -> Pro [0.1] | Det N [0.3] | N [0.5] | NP PP [0.1]
     VP -> Vi [0.05] | Vt NP [0.9] | VP PP [0.05]
     Det -> Art [1.0]
     PP -> Prep NP [1.0]
   # Lexical productions.
     Pro -> "i" [0.3] | "we" [0.1] | "you" [0.1] | "he" [0.3] | "she" [0.2]
     Art -> "a" [0.4] | "an" [0.1] | "the" [0.5]
     Prep -> "with" [0.7] | "in" [0.3]
     N -> "salad" [0.4] | "fork" [0.3] | "mushrooms" [0.3]
     Vi -> "sneezed" [0.5] | "ran" [0.5]
     Vt -> "eat" [0.2] | "eats" [0.2] | "ate" [0.2] | "see" [0.2] | "saw" [0.2]
    """)

sentence1 = "he ate salad"
sentence2 = "he ate salad with mushrooms"
sentence3 = "he ate salad with a fork"

# Un-comment the following 2 non-comment lines
# when working on `PCFG Parser` section in the lab.
## Initialize a parser with our toy probabilistic grammar
##  (it will have 'S' as the start symbol),
##  and parse a sentence
#sppc=BetterICP(grammar)
#sppc.parse(sentence1.split())

## Parse some more complex sentences
#sppc.parse(sentence2.split())
#sppc.parse(sentence3.split())
