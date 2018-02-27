from __future__ import division
import sys
from pprint import pprint
from collections import defaultdict
import nltk
from nltk.corpus import treebank
from nltk import ConditionalFreqDist, Nonterminal, FreqDist
from lab5_fix import *
from nltk import InsideChartParser
from nltk.parse.chart import Chart,AbstractChartRule
from nltk.tree import Tree,ProbabilisticTree,_child_names
from nltk.parse.pchart import ProbabilisticFundamentalRule,ProbabilisticBottomUpInitRule,ProbabilisticTreeEdge,ProbabilisticLeafEdge
from nltk.parse.pchart import SingleEdgeProbabilisticFundamentalRule
from math import log

# Renamed between 3.0 and 3.0.4 :-(
if not(hasattr(Chart,'pretty_format_edge')):
    Chart.pretty_format_edge=Chart.pp_edge

# nltk.parse.pchart is fundamentally broken, because it adds edges directly
# into the chart, where the fr can see them whether or not they've come
# out of the agenda or not.

# The least-bad fix from outside I can come up with is implemented here:
#  add a boolean var called 'pending' which is true by default, only set to
#  false when the edge comes off the agenda, and when true causes it
#  to be ignored by fr

# Possible bug?  Even pending edges _are_ checked for when testing for
#  redundancy (i.e. Chart.insert is _not_ changed), but that means any
#  failure of best-first might cause a cheaper edge to be discarded
#  because an earlier, but still pending, identical-but-more expensive
#  edge is in the chart.

nltk.chart.EdgeI.pending=True

def productions_with_left_context(self,lpos=0,leaves=None):
    """
    Generate the productions that correspond to the non-terminal nodes of the tree, with their left-context word (or None), as pairs of word and Production.
    For each subtree of the form (P: C1 C2 ... Cn) this produces a production of the
    form P -> C1 C2 ... Cn and the word to the left of C1

        >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        >>> t.productions()
        [(None, S -> NP VP),
         (None, NP -> D N),
         (None,  D -> 'the')
         ('the', N -> 'dog'),
         ('dog', VP -> V NP),
         ('dog', V -> 'chased'),
         ('dog', NP -> D N),
         ('chased', D -> 'the'),
         ('the', N -> 'cat')]

    :rtype: list(Production)
    """
    if leaves is None:
        leaves=self.leaves()
    #if not isinstance(self._label, string_types):
    #   raise TypeError('Productions can only be generated from trees having node labels that are strings')
    if lpos>0:
        lc=leaves[lpos-1]
    else:
        lc=None
    prods = [(lc,Production(Nonterminal(self._label), _child_names(self)))]
    for child in self:
        if isinstance(child, Tree):
            prods += child.productions_with_left_context(lpos,leaves)
            # could be much smarter
            lpos+=len(child.leaves())
        else:
            lpos+=1
    return prods

Tree.productions_with_left_context=productions_with_left_context

def production_distribution(psents):
    """ Creates a frequency distribution of lexical and non-lexical (grammatical) productions """
    prod_dict = defaultdict(int)
    for psent in psents:
        for production in psent.productions():
            prod_dict[production]+=1
    return prod_dict

def nt_counts(prod_dict):
    '''Create a dictionary of non-terminals and their counts'''
    nt_dict=defaultdict(int)
    for (rule,count) in prod_dict.items():
        nt_dict[rule.lhs()]+=count
    return nt_dict

def cost(prob):
    return 0.0 if prob==1.0 else -log(prob,2)

def production_cost(production,lhs_counts,production_counts):
    pcount=production_counts[production]
    ntcount=lhs_counts[production.lhs()]
    return cost(float(pcount)/float(ntcount))                     

def get_costed_productions(psents):
    """ Creates costed/weighted productions from a given list of parsed sentences."""
    prods_dict = production_distribution(psents)
    prods_nt_counts=nt_counts(prods_dict)
    costed_prods=[CostedProduction(p.lhs(),p.rhs(),production_cost(p, prods_nt_counts, prods_dict))
               for p in prods_dict.keys()]
    return costed_prods

class BetterPBPR(AbstractChartRule):
    NUM_EDGES=1
    def apply(self, chart, grammar, edge):
        if edge.is_incomplete(): return
        for prod in grammar.productions():
            if edge.lhs() == prod.rhs()[0]:
                # check for X -> X
                if prod.lhs()==edge.lhs() and len(prod.rhs())==1:
                    continue
                new_edge = ProbabilisticTreeEdge.from_production(prod, edge.start(), prod.prob())
                if chart.insert(new_edge, ()):
                    yield new_edge

class BetterSEPFR(AbstractChartRule):
    NUM_EDGES=1

    _fundamental_rule = ProbabilisticFundamentalRule()

    def apply(self, chart, grammar, edge1):
        fr = self._fundamental_rule
        if edge1.is_incomplete():
            # edge1 = left_edge; edge2 = right_edge
            for edge2 in chart.select(start=edge1.end(), is_complete=True,
                                     lhs=edge1.nextsym()):
                if edge2.pending:
                    continue
                for new_edge in fr.apply(chart, grammar, edge1, edge2):
                    yield new_edge
        else:
            # edge2 = left_edge; edge1 = right_edge
            for edge2 in chart.select(end=edge1.start(), is_complete=False,
                                      nextsym=edge1.lhs()):
                if edge2.pending:
                    continue
                for new_edge in fr.apply(chart, grammar, edge2, edge1):
                    yield new_edge

class BetterICP(InsideChartParser):
    '''Implement a more user-friendly InsideChartParser,
    which will show intermediate results, and quit after
    finding a specified number of parses'''
    def parse(self, tokens, notify=True, max=0):
        '''Run a probabilistic parse of tokens.
        If notify is true, display each complete parse as it is found
        If max>0, quit after finding that many parses'''
        self._grammar.check_coverage(tokens)
        chart = Chart(list(tokens))
        chart._trace=self._trace # Bad form. . .
        grammar = self._grammar
        start = grammar.start()
        prod_probs = {}

        # Chart parser rules.
        bu_init = ProbabilisticBottomUpInitRule()
        bu = BetterPBPR() # avoid infinite numbers of parses :-(
        fr = BetterSEPFR() # don't look at pending edges
        # Our queue
        queue = []

        # Initialize the chart.
        for edge in bu_init.apply(chart, grammar):
            if self._trace > 1:
                print('  %-50s [%.4g]' % (chart.pretty_format_edge(edge,width=2),
                                        cost(edge.prob())))
            queue.append(edge)

        found = 0
        while len(queue) > 0 and (max<1 or found<max):
            # Re-sort the queue.
            self.sort_queue(queue, chart)

            # Prune the queue to the correct size if a beam was defined
            if self.beam_size:
                self._prune(queue, chart)

            # Get the best edge.
            edge = queue.pop()
            edge.pending = False
            if self._trace > 0:
                print('  %-50s [%.4g]' % (chart.pretty_format_edge(edge,width=2),
                                        cost(edge.prob())))
            if (edge.start()==0 and
                edge.end()==chart._num_leaves and
                edge.lhs()==start and
                edge.is_complete()):
                if len(prod_probs)==0:
                    for prod in grammar.productions():
                        prod_probs[prod.lhs(), prod.rhs()] = prod.prob()
                if notify:
                    print "****"
                    for tree in chart.trees(edge, tree_class=ProbabilisticTree,
                                            complete=True):
                        self._setprob(tree, prod_probs)
                        print tree, '%.4g (%.4g)'%(cost(tree.prob()),cost(edge.prob()))
                        #print tree
                    print "****"
                found+=1
            # Apply BU & FR to it.
            queue.extend(fr.apply(chart, grammar, edge))
            queue.extend(bu.apply(chart, grammar, edge))

        # Get a list of complete parses.
        parses = list(chart.parses(grammar.start(), ProbabilisticTree))
        if not notify:
            for parse in parses:
                self._setprob(parse,prod_probs)

        # Sort by probability
        parses.sort(reverse=True, key=lambda tree: tree.prob())
        if notify:
            print "%s total parses found"%found
        return iter(parses)

    def _prune(self, queue, chart):
        """ Discard items in the queue if the queue is longer than the beam."""
        if len(queue) > self.beam_size:
            split = len(queue)-self.beam_size
            if self._trace > 2:
                for edge in queue[:split]:
                    print('  %-50s [%.4g DISCARDED]' % (chart.pretty_format_edge(edge,2),
                                                        cost(edge.prob())))
            del queue[:split]

    def beam(self,width):
        self.beam_size=width
