from __future__ import division
import sys
from types import DictType
from pprint import pprint
from collections import defaultdict
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import treebank

import rdp_fix
from rdp_fix import parse_grammar

from nltk.app import rdparser_app as rd

def production_distribution(psents):
    """ Creates a frequency distribution of lexical and non-lexical (grammatical) productions
    """
    lexdict = defaultdict(int)
    nonlexdict = defaultdict(int)
    for psent in psents:
        for production in psent.productions():
            if production.is_lexical():
                pass # students replace this
            else:
                pass # students replace this
    return lexdict,nonlexdict

#This function takes a single parsed sentence, 
#and prints the parse along with the list of all productions used in it. 
def print_parse_info(psent):
    print("\nParsed sentence:\n{}".format(psent))
    print("\nProductions used in the parse:")
    pprint(psent.productions())

def recursive_descent_parser(grammar, sentence):
    """ recursive_descent_parser takes grammar and sentence as input and 
    parses the sentence according to the grammar using recursive descent parsing technique.
    
    """
    # Loads the Recursive Descent Parser with the grammar provided
    rdp = nltk.RecursiveDescentParser(grammar, trace=2)
    # Parses the sentence and outputs a parse tree based on the grammar
    parse = rdp.parse(sentence.split())
    print parse
    return parse

def app(grammar,sent):
    """ Create a recursive descent parser demo, using a simple grammar and
    text.
    """    
    rd.RecursiveDescentApp(grammar, sent.split()).mainloop()

## Main body of code ##
# Extracting tagged sentences using NLTK libraries
# psents = treebank.parsed_sents()
# if (len(sys.argv)<2 or sys.argv[1]!='-q'):
#     try:
#         loaded+=1
#     except NameError:
#         loaded=1
#         print "\nFirst parsed sentence:\n", psents[0]
#         print "\nProductions in the first sentence:"
#         pprint(psents[0].productions())

grammar1=parse_grammar("""
    # Grammatical productions.
     S -> NP VP
     NP -> Pro | Det N | N
     Det -> Art
     VP -> V | V NP | V NP PP
     PP -> Prep NP
   # Lexical productions.
     Pro -> "i" | "we" | "you" | "he" | "she" | "him" | "her"
     Art -> "a" | "an" | "the"
     Prep -> "with" | "in"
     N -> "salad" | "fork" | "mushrooms"
     V -> "eat" | "eats" | "ate" | "see" | "saw" | "prefer" | "sneezed"
     Vi -> "sneezed" | "ran"
     Vt -> "eat" | "eats" | "ate" | "see" | "saw" | "prefer"
     Vp -> "eat" | "eats" | "ate" | "see" | "saw" | "prefer" | "gave"
    """)

sentence1 = "he ate salad"
#parse_tree = recursive_descent_parser(grammar1, sentence1)
#parse_tree.draw()
#app(grammar1, sentence1)
#sentence2 = "he ate salad with mushrooms"
#sentence3 = "he ate salad with a fork"

# Extracting tagged sentences using NLTK libraries
#psents = treebank.parsed_sents()
# print info for the 0'th parsed sentence
#print_parse_info(psents[0])

