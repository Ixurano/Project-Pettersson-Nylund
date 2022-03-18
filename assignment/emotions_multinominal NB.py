#!/usr/bin/env python
# coding: utf-8

# #### imports for the project

# In[1]:


import nltk
import random
from prettytable import PrettyTable
import textwrap 
import numpy as np
import string
from nltk.corpus import stopwords

from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier

nltk.download("punkt")
nltk.download('stopwords')


# #### Import excel with pandas Emotions

# In[2]:


import pandas as pd
df_emotions = pd.read_excel('Diabetes-classification.xlsx', sheet_name ='Emotions')

# Preparing dataset
x_emotion = df_emotions.loc[:,'discussion_text']
y_emotion = df_emotions.loc[:,'Label']

# removes all duplicates from list 
Labels_emotion = list(dict.fromkeys(y_emotion))

#Remove stopwords
lim_punc = [char for char in string.punctuation if char in "&#^_"]
nopunc = [char for char in x_emotion if char not in lim_punc]
nopunc = ''.join(nopunc)

other_stop=['•','...in','...the','...you\'ve','–','—','-','⋆','...','....','..','C.','c','|','...The','...The','...When','...A','C','+','1','2','3','4','5','6','7','8','9','10', '2016',  'speak','also', 'seen','[5].',  'using', 'get',  'instead',  "that's",  '......','may', 'e', '...it', 'puts', '...over', '[✯]','happens', "they're",'hwo',  '...a', 'called',  '50s','c;', '20',  'per', 'however,','it,', 'yet', 'one', 'bs,', 'ms,', 'sr.',  '...taking',  'may', '...of', 'course,', 'get', 'likely', 'no,']

ext_stopwords=stopwords.words('english')+other_stop
clean_words = [word for word in nopunc.split() if word.lower() not in ext_stopwords]
# puts discussion_text to a str and tokenize it
raw_text_emotion = df_emotions['discussion_text'].str.cat()

tokens_emotion = nltk.word_tokenize(raw_text_emotion)
tokens_emotion_filtered = [clean_words for clean_words in tokens_emotion if clean_words]
text_emotion = nltk.Text(tokens_emotion_filtered)


# #### Multinominal NB classifer for Emotions

# In[3]:


# the reviews will be stored as document pairs of words and category
X_list_of_words = [sentence.split(" ") for sentence in x_emotion]
documents = list(zip(X_list_of_words, y_emotion))

#give random order to the documents
random.Random(5).shuffle(documents)

tab = PrettyTable(['Discussion text', 'Emotion'])
tab.horizontal_char = '-'

for (doc, cat) in documents[0:2]:
    feats = textwrap.fill(','.join(doc[:50]), width=40)
    tab.add_row([ feats, cat])
    tab.add_row([ '\n', '\n'])
    print(cat)

print(tab)


# In[4]:


print('total words from emotion corpus: ', len(text_emotion))

# load all the words in freq distribution
all_words = nltk.FreqDist(w.lower() for w in text_emotion)

#construct a list of the 2000 most frequent words in the overall corpus (you can try with other numbers as well)
most_freq_words = all_words.most_common(6000)
print('most freq words: ', most_freq_words[100:110])

word_features = [word for (word, count) in most_freq_words]
print('word_features[:25]: ', word_features[:25])


# In[5]:


def get_document_features(document, doc_features):
    """
        This function will convert given document into a feature set.
        Note that we need to add the feature set that is relevant to the document we are inputting
        
    """
    #checking whether a word occurs in a set is much faster than checking whether it occurs in a list 
    document_words = set(document)
    features = {}
    
    #the feaures dict will consist of words as keys and boolean value of whether they exist in the document
    for word in doc_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


# test code for the above function
words_doc = text_emotion

feat_dict = get_document_features(words_doc, word_features)

feat_dict_25 = {k: feat_dict[k] for k in list(feat_dict.keys())[:25]}
print('transformed document features, printing the first 25 features \n\n', feat_dict_25)


# In[6]:


#obtain feature set
featuresets = [(get_document_features(d,word_features), c) for (d,c) in documents]

#split into train and test set (you can experiment with distribution here) 100 - 100 og
train_set, test_set = featuresets[100:1000], featuresets[:100]

#instantiate classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

#print accuracy and most informative features
print(nltk.classify.accuracy(classifier, test_set)) 

classifier.show_most_informative_features(20)


# In[7]:


from collections import defaultdict
refsets = defaultdict(set)
testsets = defaultdict(set)
labels = []
tests = []
for i, (feats, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)
    labels.append(label)
    tests.append(observed)

print(nltk.ConfusionMatrix(labels, tests))


# In[8]:



sample_review = "My sickness got worse, and the doctors won't do anything"

#get features specific to the input text
sample_features = {word:True for word in sample_review.split()}

sample_review_doc_feats = get_document_features(sample_review.split(),sample_features)


print('result of sample review: ', classifier.classify(sample_review_doc_feats))

