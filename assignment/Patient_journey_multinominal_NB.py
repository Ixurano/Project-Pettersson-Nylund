#!/usr/bin/env python
# coding: utf-8

# #### Imports
# 

# In[5]:


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

# In[6]:


import pandas as pd
df_patient = pd.read_excel('Diabetes-classification.xlsx', sheet_name='Patient-journey') # Reads in excel

# Preparing dataset
x_journey = df_patient.loc[:,'discussion_text']
y_journey = df_patient.loc[:,'Label']
# removes all duplicates from list 
Labels_journey = list(dict.fromkeys(y_journey)) 
#stopwords
lim_punc_patient = [char for char in string.punctuation if char in "&#^_"]
nopunc_patient = [char for char in x_journey if char not in lim_punc_patient]
nopunc_patient = ''.join(nopunc_patient)

other_stop=['•','...in','...the','...you\'ve','–','—','-','⋆','...','....','..','C.','c','|','...The','...The','...When','...A','C','+','1','2','3','4','5','6','7','8','9','10', '2016',  'speak','also', 'seen','[5].',  'using', 'get',  'instead',  "that's",  '......','may', 'e', '...it', 'puts', '...over', '[✯]','happens', "they're",'hwo',  '...a', 'called',  '50s','c;', '20',  'per', 'however,','it,', 'yet', 'one', 'bs,', 'ms,', 'sr.',  '...taking',  'may', '...of', 'course,', 'get', 'likely', 'no,']

ext_stopwords_patient=stopwords.words('english')+other_stop
clean_words = [word for word in nopunc_patient.split() if word.lower() not in ext_stopwords_patient]

# puts discussion_text to a str and tokenize it
raw_text_journey = df_patient['discussion_text'].str.cat()
tokens_journey = nltk.word_tokenize(raw_text_journey)
tokens_emotion_filtered = [clean_words for clean_words in tokens_journey if clean_words.isalnum()]
text_journey = nltk.Text(tokens_journey)


# #### Loads in Patient Journey labels

# In[7]:


import pandas as pd
df_patient = pd.read_excel('Diabetes-classification.xlsx', sheet_name='Patient-journey') # Reads in excel

# Preparing dataset
x_journey = df_patient.loc[:,'discussion_text']
y_journey = df_patient.loc[:,'Label']
# removes all duplicates from list 
Labels_journey = list(dict.fromkeys(y_journey)) 
#stopwords
lim_punc_patient = [char for char in string.punctuation if char in "&#^_"]
nopunc_patient = [char for char in x_journey if char not in lim_punc_patient]
nopunc_patient = ''.join(nopunc_patient)
ext_stopwords_patient=stopwords.words('english')+other_stop
clean_words = [word for word in nopunc_patient.split() if word.lower() not in ext_stopwords_patient]

# puts discussion_text to a str and tokenize it
raw_text_journey = df_patient['discussion_text'].str.cat()
tokens_journey = nltk.word_tokenize(raw_text_journey)
tokens_emotion_filtered = [clean_words for clean_words in tokens_journey if clean_words.isalnum()]
text_journey = nltk.Text(tokens_journey)


# In[8]:


# the reviews will be stored as document pairs of words and category
X_list_of_words_journey = [sentence.split(" ") for sentence in x_journey]
documents_journey = list(zip(X_list_of_words_journey, y_journey))

#give random order to the documents
random.shuffle(documents_journey)

tab = PrettyTable(['Discussion text', 'Emotion'])
tab.horizontal_char = '-'

for (doc, cat) in documents_journey[0:2]:
    feats_journey = textwrap.fill(','.join(doc[:50]), width=40)
    tab.add_row([ feats_journey, cat])
    tab.add_row([ '\n', '\n'])
    print(cat)

print(tab)


# In[9]:


print('total words from emotion corpus: ', len(text_journey))

# load all the words in freq distribution
all_words_journey = nltk.FreqDist(w.lower() for w in text_journey)

#construct a list of the 2000 most frequent words in the overall corpus (you can try with other numbers as well)
most_freq_words_journey = all_words_journey.most_common(6000)
print('most freq words: ', most_freq_words_journey[100:110])

word_features_journey = [word for (word, count) in most_freq_words_journey]
print('word_features[:25]: ', word_features_journey[:25])


# In[12]:


def get_document_features_journey(documents_journey, doc_features):
    """
        This function will convert given document into a feature set.
        Note that we need to add the feature set that is relevant to the document we are inputting
        
    """
    #checking whether a word occurs in a set is much faster than checking whether it occurs in a list 
    document_words = set(documents_journey)
    features = {}
    
    #the feaures dict will consist of words as keys and boolean value of whether they exist in the document
    for word in doc_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

words_doc = text_journey

feat_dict = get_document_features_journey(words_doc, word_features_journey)

feat_dict_25 = {k: feat_dict[k] for k in list(feat_dict.keys())[:25]}
print('transformed document features, printing the first 25 features \n\n', feat_dict_25)


# In[13]:


#obtain feature sets for all movie reviews
featuresets_journey = [(get_document_features_journey(d,word_features_journey), c) for (d,c) in documents_journey]

#split into train and test set (you can experiment with distribution here) 100 - 100 og
train_set_journey, test_set_journey = featuresets_journey[300:3000], featuresets_journey[:300]

#instantiate classifier
classifier = nltk.NaiveBayesClassifier.train(train_set_journey)

#print accuracy and most informative features
print(nltk.classify.accuracy(classifier, test_set_journey)) 

classifier.show_most_informative_features(20)


# In[14]:


from collections import defaultdict
refsets = defaultdict(set)
testsets = defaultdict(set)
labels = []
tests = []
for i, (feats, label) in enumerate(test_set_journey):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)
    labels.append(label)
    tests.append(observed)

print(nltk.ConfusionMatrix(labels, tests))


# In[ ]:


sample_review = "My doctor told me to start running and go on a diet"

#get features specific to the input text
sample_features = {word:True for word in sample_review.split()}


sample_review_doc_feats = get_document_features_journey(sample_review.split(),sample_features)


#print('Sample review features: \n\n',sample_review_doc_feats)

print('result of sample review: ', classifier.classify(sample_review_doc_feats))

