import re
import operator
import os

answers_test_set = dict({
  'F-test1': 'F', 
  'F-test2': 'F', 
  'M-test3': 'M', 
  'F-test4': 'F', 
  'F-test5': 'F', 
  'F-test6': 'F', 
  'M-test7': 'M', 
  'M-test8': 'M', 
  'M-test9': 'm',
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


def text_normalizer(raw_text):
  """Return a normalized text string.""" 

  # normalizing abbreviations
  raw_text = raw_text.replace("'s", " is")
  raw_text = raw_text.replace("'m", " am")
  raw_text = raw_text.replace("'ve", " have")
  raw_text = raw_text.replace("'ll", " will")
  raw_text = raw_text.replace("'nt", " not")
  #text = text.replace("'d", " would")

  # replacing the new line symbol with a space
  raw_text = raw_text.replace("\n", " ")
  raw_text = raw_text.replace("(", " ")
  raw_text = raw_text.replace(")", " ")

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

  result = re.split(r' ', text)
  return result


def add_and_count(tokens, dict_name):
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


def occurrence_counter(sorted_list, occurence):
  """Counts how many words there are for a specific number of occurences."""

  counter = 0

  for word in sorted_list:
    if word[1] == occurence:
      counter += 1

  print 'there are ' + str(counter) + ' words that occur ' + str(occurence) + ' times'

  return counter


# path of current training data directory
path_train_data = os.path.dirname(os.path.abspath(__file__)) + '/train/'
train_files = os.walk(path_train_data)


# total collection of gender-specific words
female_total_count = dict()
male_total_count = dict()
combined_total_count = dict()


# iterate through training data and add words to collection
for filenames in train_files:
  for filename in filenames[2]:

    f = open('train/' + filename, 'r')
    text = f.read()
    text = text_normalizer(text)
    tokens = text_tokenizer(text)

    add_and_count(tokens, combined_total_count)

    if filename[0] == 'F':
      # add tokens to global female dict
      add_and_count(tokens, female_total_count)
      #print filename + ' ' + 'female'
    else:
      # add tokens to global male dict
      add_and_count(tokens, male_total_count)
      #print filename + ' ' + 'male'




# take counted tokens and sort them by frequency in reverse order
female_total_count_sorted = sorted(female_total_count.items(), key=operator.itemgetter(1), reverse=True)
male_total_count_sorted = sorted(male_total_count.items(), key=operator.itemgetter(1), reverse=True)
combined_total_count_sorted = sorted(combined_total_count.items(), key=operator.itemgetter(1), reverse=True)

# print length of different vocubularies
print 'combined total vocabulary length:'
print len(combined_total_count_sorted)
print '\n'
print 'female total vocabulary length:'
print len(female_total_count_sorted)
print '\n'
print 'male total vocabulary length:'
print len(male_total_count_sorted)
print '\n'


# print occurences of rare words
occurrence_counter(combined_total_count_sorted, 4)
occurrence_counter(combined_total_count_sorted, 3)
occurrence_counter(combined_total_count_sorted, 2)
occurrence_counter(combined_total_count_sorted, 1)
#occurrence_counter(female_total_count_sorted, 1)
#occurrence_counter(male_total_count_sorted, 1)


# strip low frequency words
female_total_count_sorted = strip_below(female_total_count_sorted, 25)
male_total_count_sorted = strip_below(male_total_count_sorted, 25)
#combined_total_count_sorted = strip_below(combined_total_count_sorted, 2)

# delete from certain index on 
#del female_total_count_sorted[250:]
#del male_total_count_sorted[250:]
del combined_total_count_sorted[10:]

print 'females top words:'
print female_total_count_sorted
print '\n'
print 'males top words:'
print male_total_count_sorted
print '\n'
print 'top 10 words combined:'
print combined_total_count_sorted




