import ipdb

import nltk

#import brown corpus
from nltk.corpus import brown

# module for training a Hidden Markov Model and tagging sequences
from nltk.tag.hmm import HiddenMarkovModelTagger

# module for computing a Conditional Frequency Distribution
from nltk.probability import ConditionalFreqDist

# module for computing a Conditional Probability Distribution
from nltk.probability import ConditionalProbDist

# module for computing a probability distribution with the Maximum Likelihood Estimate
from nltk.probability import MLEProbDist

import operator
import random
import numpy as np

############# INTRO POS #################

def intro():
  # NLTK provides corpora tagged with part-of-speech (POS) information and some tools to access this information
  # The Penn Treebank tagset is commonly used for English
  nltk.help.upenn_tagset()

  # We can retrieve the tagged sentences in the Brown corpus by calling the tagged_sents() function
  tagged_sentences = brown.tagged_sents(categories= 'news')
  print "Sentence tagged with Penn Treebank POS labels:"
  print tagged_sentences[42]

   # We can access the Universal tags by changing the tagset argument
  tagged_sentences_universal = brown.tagged_sents(categories= 'news', tagset='universal')
  print "Sentence tagged with Universal POS:"
  print tagged_sentences_universal[42]

# Comment to hide intro
intro()


############# EXERCISE 1 #################
# Solution for exercise 1
# Input: genre (string), tagset (string)
# Output: number_of_tags (int), top_tags (list of string)


# get the number of tags found in the corpus
# compute the Frequency Distribution of tags

def ex1(genre,tagset):
  
  # get the tagged words from the corpus
  tagged_words = brown.tagged_words(categories= genre, tagset=tagset)
  

  # TODO: build a list of the tags associated with each word
  tags = brown.tagged_sents(categories=genre, tagset=tagset)
  
  # TODO: using the above list compute the Frequency Distribution of tags in the corpus
  # hint: use nltk.FreqDist()
  tagsFDist = nltk.FreqDist([tag for sentence in tags for term, tag in sentence])

  
  # TODO: retrieve the total number of tags in the tagset
  number_of_tags = len(tagsFDist.keys())
  
  #TODO: retrieve the top 10 most frequent tags
  top_tags = [tag for tag, freq in sorted(tagsFDist.iteritems(), key=operator.itemgetter(1), reverse=True)[:10]]

  return (number_of_tags,top_tags)



def test_ex1():
  print "Tag FreqDist for news:"
  print ex1('news',None)

  print "Tag FreqDist for science_fiction:"
  print ex1('science_fiction',None)

  # Do the same thing for a different tagset: Universal

  print "Tag FreqDist for news with Universal tagset:"
  print ex1('news','universal')

  print "Tag FreqDist for science_fiction with Universal tagset:"
  print ex1('science_fiction','universal')

### Uncomment to test exerise 1
# Let's look at the top tags for different genre and tagsets
#  and observe the differences
# test_ex1()

############# EXERCISE 2 #################
# Solution for exercise 2
# Input: sentence (list of string), size (<4600)
# Output: hmm_tagged_sentence (list of tuples), tagger (HiddenMarkovModelTagger)

# hint: use the help on HiddenMarkovModelTagger to find out how to train, tag and evaluate the HMM tagger
def ex2(sentence, size):
  tagged_sentences = brown.tagged_sents(categories= 'news')
  
  # set up the training data
  train_data = tagged_sentences[-size:]
  
  # set up the test data
  test_data = tagged_sentences[:100]

  # TODO: train a HiddenMarkovModelTagger, using the train() method
  tagger = nltk.tag.hmm.HiddenMarkovModelTagger.train(train_data)

  # TODO: using the hmm tagger tag the sentence
  hmm_tagged_sentence = tagger.tag_sents([sentence])[0]
  
  # TODO: using the hmm tagger evaluate on the test data
  eres = tagger.evaluate(test_data)

  return (tagger, hmm_tagged_sentence, eres)


def test_ex2():
  tagged_sentences = brown.tagged_sents(categories= 'news')
  words = [tp[0] for tp in tagged_sentences[42]]
  (tagger, hmm_tagged_sentence, eres ) = ex2(words,500)
  print "Sentenced tagged with nltk.HiddenMarkovModelTagger:"
  print hmm_tagged_sentence
  print "Eval score:"
  print eres

  (tagger, hmm_tagged_sentence, eres ) = ex2(words,3000)
  print "Sentenced tagged with nltk.HiddenMarkovModelTagger:"
  print hmm_tagged_sentence
  print "Eval score:"
  print eres

### Uncomment to test exerise 2
#Look at the tagged sentence and the accuracy of the tagger. How does the size of the training set affect the accuracy?
# test_ex2()



############# EXERCISE 3 #################
# Solution for exercise 3
# Input: tagged_words (list of tuples)
# Output: emission_FD (ConditionalFreqDist), emission_PD (ConditionalProbDist), p_NN (float), p_DT (float)


# in the previous labs we've seen how to build a freq dist
# we need conditional distributions to estimate the transition and emission models
# in this exerise we estimate the emission model
def ex3(tagged_words):

  # TODO: prepare the data
  # the data object should be a list of tuples of conditions and observations
  # in our case the tuples will be of the form (tag,word) where words are lowercased
  data = [ (tag, word.lower()) for word, tag in tagged_words]


  # TODO: compute a Conditional Frequency Distribution for words given their tags using our data
  emission_FD = nltk.probability.ConditionalFreqDist(data)
  
  # TODO: return the top 10 most frequent words given the tag NN
  top_NN = []
  if emission_FD.get('NN') is not None:
    top_NN = sorted(emission_FD.get('NN').iteritems(), key=operator.itemgetter(1), reverse=True)[:10]

  # TODO: Compute the Conditional Probability Distribution using the above Conditional Frequency Distribution. Use MLEProbDist estimator.
  emission_PD = nltk.probability.ConditionalProbDist(emission_FD, MLEProbDist)
  
  # TODO: compute the probabilities of P(year|NN) and P(year|DT)
  p_NN = emission_PD.get('NN').prob('year') if emission_PD.get('NN') is not None else 0.0
  p_DT = emission_PD.get('DT').prob('year') if emission_PD.get('DT') is not None else 0.0
  
  return (emission_FD, top_NN, emission_PD, p_NN, p_DT)


def test_ex3():
  tagged_words = brown.tagged_words(categories='news')
  (emission_FD, top_NN, emission_PD, p_NN, p_DT) = ex3(tagged_words)
  print "Frequency of words given the tag *NN*: ", top_NN
  print "P(year|NN) = ", p_NN
  print "P(year|DT) = ", p_DT

### Uncomment to test exerise 3
#Look at the estimated probabilities. Why is P(year|DT) = 0 ? What are the problems with having 0 probabilities and what can be done to avoid this?
# test_ex3()

############# EXERCISE 4 #################
# Solution for exercise 4
# Input: tagged_sentences (list)
# Output: transition_FD (ConditionalFreqDist), transition_PD (ConditionalProbDist), p_VBD_NN, p_DT_NN

# compute the transition probabilities
# the probabilties of a tag at position i+1 given the tag at position i
def ex4(tagged_sentences):
  
  # TODO: prepare the data
  # the data object should be an array of tuples of conditions and observations
  # in our case the tuples will be of the form (tag_(i),tag_(i+1))
  data = [(sentence[i][1], sentence[i+1][1])  for sentence in tagged_sentences for i  in range(len(sentence)-1)]
  data = [("<S>", sentence[0][1]) for sentence in tagged_sentences] + data
  # TODO: compute the Conditional Frequency Distribution for a tag given the previous tag
  transition_FD = nltk.probability.ConditionalFreqDist(data)
  
  # TODO: compute the Conditional Probability Distribution for the
  # transition probability P(tag_(i+1)|tag_(i)) using the MLEProbDist
  # to estimate the probabilities
  transition_PD = nltk.probability.ConditionalProbDist(transition_FD, MLEProbDist)

  # TODO: compute the probabilities of P(NN|VBD) and P(NN|DT)
  p_VBD_NN = transition_PD.get('VBD').prob('NN') if transition_PD.get('VBD') is not None else 0.0
  p_DT_NN = transition_PD.get('DT').prob('NN') if transition_PD.get('DT') is not None else 0.0

  return (transition_FD, transition_PD,p_VBD_NN, p_DT_NN )


def test_ex4():
  tagged_sentences = brown.tagged_sents(categories= 'news')
  (transition_FD, transition_PD,p_VBD_NN, p_DT_NN ) = ex4(tagged_sentences)
  print "P(NN|VBD) = ", p_VBD_NN
  print "P(NN|DT) = ", p_DT_NN

### Uncomment to test exerise 4
# Are the results what you would expect? The sequence NN DT seems very probable. How will this affect the sequence tagging?
# test_ex4()


############# EXERCISE 5 #################
# Solution for exercise 5
# Input: sentence (list), tagset (list), emission_PD (ConditionalProbDist), transition_PD (ConditionalProbDist)
# Output: best state sequence, probability (or cost) or best state sequence  

def viterbi(sentence, states, emission_PD, transition_PD):
  #TODO: compute the sentence length
  T = len(sentence)
  #TODO: compute the number of possible tags (from tagset)
  N = len(states)

  #TODO: create the dynamic program table (TxN array)
  V = np.zeros((T, N), np.float64)

  #TODO: create the backpointers dictionnary 
  B = {}

  #TODO: find the starting log probabilities for each state
  starting_probs = transition_PD.get("<S>")
  scores_0 = [starting_probs.logprob(state) + emission_PD.get(state).logprob(sentence[0].lower()) for state in states]

  #TODO: find the maximum log probabilities for reaching each state at time t
  all_scores = [ scores_0 ]
  for t in range(1, T):
    print t
    print len(all_scores)
    scores = [0.0] * N
    for i, state in enumerate(states):
      # CONTINUE TOMORROW
      partial_scores = [transition_PD.get(states[prev_i]).logprob(state) + emission_PD.get(state).logprob(sentence[t]) + all_scores[t-1][prev_i] for prev_i in range(N)]
      max_index = np.argmax(partial_scores)
      B[(t, state)] = states[max_index]
      scores[i] = partial_scores[max_index]
    all_scores.append(scores)
   
  #TODO: find the highest probability final state
  final_state = states[np.argmax(all_scores[-1])]

  sequence = [ 'NONE' ] * T    
  #TODO: traverse the back-pointers B to find the state sequence
  sequence[T-1] = final_state
  for t in range(0, T-1)[::-1]:
    sequence[t] = B.get((t+1, sequence[t+1]))
  ipdb.set_trace()

  return sequence


def test_ex5():
  tagged_words = brown.tagged_words(categories='news', tagset='universal')
  (emission_FD, top_NN, emission_PD, p_NN, p_DT) = ex3(tagged_words)
  tagged_sentences = brown.tagged_sents(categories= 'news', tagset='universal')
  (transition_FD, transition_PD, p_VBD_NN, p_DT_NN ) = ex4(tagged_sentences)
  states = list(set(pos for (word,pos) in brown.tagged_words(categories='news', tagset='universal')))
  sentence = [tp[0] for tp in tagged_sentences[42]]
  tag_sequence = viterbi(sentence, states, emission_PD, transition_PD)
  print "Viterbi tag sequence:" + ' '.join(tag_sequence)
  print "Gold tag sequence:" + ' '.join([tp[1] for tp in tagged_sentences[42]])

### Uncomment to test exercise 5
test_ex5()