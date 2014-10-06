# import modules
import re
import operator
import os
import math

"""config global variables"""
# Simple Laplace smoothing
smoothing = 4
# strip vocabulary below this number of occurences
strip_below_threshold = 15
# work with which kind of ngrams
ngram = 'unigram'
#ngram = 'bigram'
#ngram = 'trigram'

# this is just for determining accuracy of results
answers_test_set = dict({
  'F-test1': 'F', 
  'F-test2': 'F', 
  'M-test3': 'M', 
  'F-test4': 'F', 
  'F-test5': 'F', 
  'F-test6': 'F', 
  'M-test7': 'M', 
  'M-test8': 'M', 
  'M-test9': 'M',
  'F-test10': 'F',
  'M-test11': 'M',
  'M-test12': 'M',
  'M-test13': 'M',
  'F-test14': 'F',
  'F-test15': 'F',
  'M-test16': 'M',
  'F-test17': 'F',
  'F-test18': 'F',
  'F-test19': 'F',
  'F-test20': 'F',
  'F-test21': 'F',
  'F-test22': 'F',
  'F-test23': 'F',
  'F-test24': 'F',
  'M-test25': 'M',
  'M-test26': 'M',
  'M-test27': 'M',
  'M-test28': 'M',
  'M-test29': 'M',
  'F-test30': 'F',
  'M-test31': 'M',
  'F-test32': 'F',
  'M-test33': 'M',
  'M-test34': 'M',
  'F-test35': 'F',
  'M-test36': 'M',
  'M-test37': 'M',
  'M-test38': 'M',
  'M-test39': 'M',
  'F-test40': 'F',
  'M-test41': 'M',
  'F-test42': 'F',
  'M-test43': 'M',
  'M-test44': 'M',
  'F-test45': 'F',
  'M-test46': 'M',
  'M-test47': 'M',
  'F-test48': 'F',
  'F-test49': 'F',
  'F-test50': 'F',
  })


#following are the function definitions

def text_normalizer(raw_text):
  """Return a normalized text string.""" 

  # normalizing abbreviations
  raw_text = raw_text.replace("'s", " is")
  raw_text = raw_text.replace("'m", " am")
  raw_text = raw_text.replace("'ve", " have")
  raw_text = raw_text.replace("'ll", " will")
  raw_text = raw_text.replace("'t", " not")
  raw_text = raw_text.replace("cant", "can not")
  #raw_text = raw_text.replace("I'd", "I would")
  #text = text.replace("'d", " would")

  # replacing the new line symbol with a space
  raw_text = raw_text.replace("\n", " ")
  raw_text = raw_text.replace("(", " ")
  raw_text = raw_text.replace(")", " ")
  raw_text = raw_text.replace("[", " ")
  raw_text = raw_text.replace("]", " ")

  # transform all words to lowercase
  raw_text = raw_text.lower()

  normalized_text = raw_text
  return normalized_text


def text_tokenizer(text):
  """Return an array of separate tokens.""" 

  # get rid of sentence delimiters
  text = re.sub(r'(\. )|(\.\.)|(!)|(\?)|(,)|(;)|(:)|(\- )|(\')|(\")', ' ', text)

  # replace occurences of double spaces
  text = re.sub(' +',' ', text)

  unigram = re.split(r' ', text)
  bigram = []
  trigram = []

  # build set of bigrams based on existing unigrams
  if ngram == 'bigram':
    for idx, word in enumerate(unigram):
      if idx < len(unigram) - 2:
        item = word + ' ' + unigram[idx + 1]
        bigram.append(item)
    return bigram

  if ngram == 'trigram':
    for idx, word in enumerate(unigram):
      if idx < len(unigram) - 3:
        item = word + ' ' + unigram[idx + 1] + ' ' + unigram[idx + 2]
        trigram.append(item)
    return trigram

  else:
    return unigram


def count_tokens_and_add(tokens, dict_name):
  """Takes a set of tokens, counts them and adds them to a dictionary."""

  # count occurences of words in tokens array
  # and add them to the corresponding dict if hash doesn't exist yet
  for token in tokens:
    if token not in dict_name:
      new_word = dict({token: 1})
      dict_name.update(new_word)
    else:
      dict_name[token] += 1

  return dict_name


def count_total(vocabulary):
  """
  Counts how many occurences there are in total.
  vocabulary must be a hashable dictionary, not a simple list.
  """

  total_count = 0
  for word in vocabulary:
    total_count = vocabulary[word] + total_count

  return total_count


# this function only works with a list
def strip_below(sorted_list, threshold):
  """Strips a list below a certain threshold."""

  index = 0
  threshold -= 1
  while sorted_list[index][1] != threshold:
    index += 1

  # print how many words there are below the threshold
  print 'there are ' + str(index) + ' words that occur ' + str(threshold + 1) + ' times or more'

  del sorted_list[index:]
  return sorted_list


# this function only works with a dict
def strip_dict_below(voc_dict, threshold):
  """Strips entries in a dict that are below a certain threshold."""

  new_dict = voc_dict.copy()

  for word in voc_dict:
    if voc_dict[word] < threshold:
      new_dict.pop(word)

  return new_dict


def occurrence_counter(sorted_list, occurence):
  """Counts how many words there are for a specific number of occurences."""

  counter = 0

  for word in sorted_list:
    if word[1] == occurence:
      counter += 1

  print 'there are ' + str(counter) + ' words that occur ' + str(occurence) + ' times'

  return counter


def calc_single_word_prob(word, dict_vocabulary):
  """Calculates the likelihood of a single word occuring in the given text."""

  total = count_total(dict_vocabulary)

  # if word does not exist in dictionary apply some smoothing
  if word in dict_vocabulary:
    word_occ = dict_vocabulary[word]
    prob = float(word_occ) / total
  else:
    # smoothing
    word_occ = smoothing 
    # add smoothing to total word count
    prob = float(word_occ) / (total + smoothing) 
  
  # use logarithm to prevent underflow later
  prob_log = math.log(prob, 10)

  return prob_log


def calc_total_prob(word_list, dict_vocabulary):
  """
  Calculates the probability of a certain text being written by male or female
  based on the given vocabulary.
  """

  total_prob = 0

  for word in word_list:
    word_prob = calc_single_word_prob(word, dict_vocabulary)
    total_prob += word_prob

  return total_prob


def iterate_test_set(test_set, female_voc, male_voc):
  """"
  Iterate through test set and calculate probabilites 
  for either being male or female and compare them
  """

  right_answers = 0

  # table head
  print 'Iterating through text files ...'
  print '+----------------+---------------+---------------+---------------+'
  print '| file name \t | guess \t | answer \t | y/n \t\t |'
    
  for test_doc in answers_test_set:

    answer = ''

    f = open('test/' + test_doc + '.txt', 'r')
    text = f.read()
    text = text_normalizer(text)
    tokens = text_tokenizer(text)

    female_prob = calc_total_prob(tokens, female_voc)
    male_prob = calc_total_prob(tokens, male_voc)

    print '+----------------+---------------+---------------+---------------+'
    
    # since the probability is a negative log the greater the number
    # the higher the probability
    if female_prob > male_prob:
      row = '| ' + test_doc + '\t | female \t | ' + answers_test_set[test_doc] + '\t\t | '
      answer = 'F'
    else:
      row = '| ' + test_doc + '\t | male \t | ' + answers_test_set[test_doc] + '\t\t | '
      answer = 'M'

    if answer == answers_test_set[test_doc]:
      right_answers += 1
      row += 'correct \t |'
    else:
      row += 'false \t |'

    print row

  print '+----------------+---------------+---------------+---------------+'
  percentage_correct = float(right_answers) / len(answers_test_set) * 100
  print '----> ' + str(percentage_correct) + '% of the answers were correct'
  print '\n'

  return percentage_correct


def iterate_training_set(train_data_folder):
  """"
  Iterate through test set and calculate probabilites 
  for either being male or female and compare them
  """

  # path of current training data directory
  path_train_data = os.path.dirname(os.path.abspath(__file__)) + '/' + train_data_folder + '/'
  train_files = os.walk(path_train_data)

  # total collection of gender-specific words
  word_count = dict()
  word_count['female'] = dict()
  word_count['male'] = dict()
  word_count['combined'] = dict()

  # iterate through training data and add words to collection
  for filenames in train_files:
    for filename in filenames[2]:

      f = open(train_data_folder + '/' + filename, 'r')
      text = f.read()
      text = text_normalizer(text)
      tokens = text_tokenizer(text)

      count_tokens_and_add(tokens, word_count['combined'])

      if filename[0] == 'F':
        # add tokens to global female dict
        count_tokens_and_add(tokens, word_count['female'])
        #print filename + ' ' + 'female'
      else:
        # add tokens to global male dict
        count_tokens_and_add(tokens, word_count['male'])
        #print filename + ' ' + 'male'

  return word_count


def print_top(voc, threshold, voc_type):
  """Takes the top words of the vocabulary and prints them in a nice format."""

  del voc[threshold:]
  index = 1

  print 'These are the top ' + str(threshold) + ' words in the ' + voc_type + ' vocabulary:'
  
  for key in voc:
    print '+--------+---------------+---------------+'
    print '| ' + str(index) + '\t | ' + key[0] + '\t\t | ' + str(key[1]) + '\t |'
    index += 1

  print '+--------+---------------+---------------+'
  print '\n'

  # return stripped voc
  return voc



# iterate through training set and return counted words dict
word_count = iterate_training_set('train')

# strip word count of infrequently occuring words
female_total_count_stripped = strip_dict_below(word_count['female'], strip_below_threshold)
male_total_count_stripped = strip_dict_below(word_count['male'], strip_below_threshold)
combined_total_count_stripped = strip_dict_below(word_count['combined'], strip_below_threshold)

# take counted tokens and sort them by frequency in reverse order
female_total_count_sorted = sorted(female_total_count_stripped.items(), key=operator.itemgetter(1), reverse=True)
male_total_count_sorted = sorted(male_total_count_stripped.items(), key=operator.itemgetter(1), reverse=True)
combined_total_count_sorted = sorted(combined_total_count_stripped.items(), key=operator.itemgetter(1), reverse=True)

# print length of different vocabularies
print '\n'
print 'Occurences of unique words:'
print '+----------------+---------------+---------------+'
print '| combined \t | female \t | male \t |'
print '+----------------+---------------+---------------+'
print '| ' + str(len(word_count['combined'])) + ' words \t | ' + str(len(word_count['female'])) + ' words \t | ' + str(len(word_count['male'])) + ' words \t |'
print '+----------------+---------------+---------------+'
print '\n'


# print occurences of rare words
#occurrence_counter(combined_total_count_sorted, 4)
#occurrence_counter(combined_total_count_sorted, 3)
#occurrence_counter(combined_total_count_sorted, 2)
#occurrence_counter(combined_total_count_sorted, 1)
#occurrence_counter(female_total_count_sorted, 1)
#occurrence_counter(male_total_count_sorted, 1)


# print the top words for female, male and combined
comb_top_voc = print_top(combined_total_count_sorted, 5, 'combined')
female_top_voc = print_top(female_total_count_sorted, 5, 'female')
male_top_voc = print_top(male_total_count_sorted, 5, 'male')


# receive gender-specific words by comparing vocabularies
# somehow this only works for the first loop, the deletion process does not work 
# for the second loop, I couldn't figure out why
"""male_charac = male_top_voc

for female_word in female_top_voc:
  index = 0
  for male_word in male_top_voc:
    if male_word[0] == female_word[0]:
      del male_charac[index]
    index += 1

print 'characteristic male words'
print male_charac


female_charac = female_top_voc

for male_word in male_top_voc:
  index = 0
  for fe_word in female_top_voc:
    if fe_word[0] == male_word[0]:
      del female_charac[index]
    index += 1

print 'characteristic female words'
print female_charac"""


# do some magic
iterate_test_set(answers_test_set, female_total_count_stripped, male_total_count_stripped)

