#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install spacy')

get_ipython().system('python -m spacy download en')


# In[2]:


# Word toeknization
from spacy.lang.en import English

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()

text = """When learning data science, you shouldn't get discouraged!
Challenges and setbacks aren't failures, they're just part of the journey. You've got this!"""

# "nlp" Object is used to create documents with linguistic annotations.
my_doc = nlp(text)

# Create list of word tokens
token_list = []
for token in my_doc:
    token_list.append(token.text)
print(token_list)


# In[4]:


# sentence tokenization

# Load English tokenizer, tagger, parser, NER and word vecctors
nlp = English()

# Create the pipeline 'sentencizer' component
sbd = nlp.create_pipe('sentencizer')

# Add the component to the pipeline
nlp.add_pipe('sentencizer')

text = """When learning data science, you shouldn't get discouraged!
Challenges and setbacks aren't failures, they're just part of the journey. You've got this!"""

# "nlp" Object is used to create documents with linguistic annotations.
doc = nlp(text)

# create list of sentence tokens
sents_list = []
for sent in doc.sents:
    sents_list.append(sent.text)
print(sents_list)


# In[5]:


#Stop words
#importing stop words from English language.
import spacy
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

#Printing the totasl number of stop words:
print('Number of stop words: %s' %len(spacy_stopwords))

#Printing first ten stop words:
print('First ten stop words: %s' % list(spacy_stopwords)[:20])


# In[7]:


from spacy.lang.en.stop_words import STOP_WORDS

#Implementation of stop words:
filtered_sent=[]

# "nlp" Object is used to create documents with linguistic annotations.
doc = nlp(text)

#filtering stop words
for word in doc:
    if word.is_stop==False:
        filtered_sent.append(word)
print("Filtered Sentence:",filtered_sent)


# In[11]:


# Implementation lemmatization
sentence = nlp("run runs running runner")

# finding lemma for each word
for word in sentence:
    print(word.text, word.lemma_)


# In[12]:


# POS tagging

# importing the model en_core_web_sm of English for vocabulary, syntax and entities
import en_core_web_sm

# load en_core_web_sm of English for vocabulary, syntax & entites
nlp = en_core_web_sm.load()

# "nlp" Object is used to create documents with linguistic annotations
docs = nlp(u"All is well that ends well.")

for word in docs:
    print(word.text,word.pos_)


# In[14]:


#for visualization of Entity detection importing displacy from spacyt:

from spacy import displacy

nytimes = nlp(u"""New York City on Tuesday declared a public health emergency and ordered mandatory measles vaccinations amid an outbreak, becoming the latest national flash point over refusals to inoculate against dangerous diseases.

At least 285 people have contracted measles in the city since September, mostly in Brooklyn’s Williamsburg neighborhood. The order covers four Zip codes there, Mayor Bill de Blasio (D) said Tuesday.

The mandate orders all unvaccinated people in the area, including a concentration of Orthodox Jews, to receive inoculations, including for children as young as 6 months old. Anyone who resists could be fined up to $1,000.""")

entities=[(i, i.label_, i.label) for i in nytimes.ents]
entities


# In[17]:


displacy.render(nytimes, style = "ent")


# In[19]:


docp = nlp("In pursuit of wall, President Trump ran into one.")

for chunk in docp.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)


# In[20]:


displacy.render(docp, style="dep")


# In[21]:


import en_core_web_sm
nlp = en_core_web_sm.load()
mango = nlp(u'mango')
print(mango.vector.shape)
print(mango.vector)


# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline


# In[ ]:


get_ipython().system('pip uninstall scikit-learn')


# In[ ]:


get_ipython().system('pip install scikit-learn==0.13')


# In[ ]:


# Loading TSV file
df = pd.DataFrame
df_amazon = pd.read_csv ("datasets/amazon_alexa.tsv", sep="\t")


# In[ ]:


# Top 5 records
df.head()


# In[ ]:


# shape of dataframe
df_amazon.shape


# In[ ]:


# View data information
df_amazon.info()


# In[ ]:


import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Crearing our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations
    mytokens = parser(sentence)
    
    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    
    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    
    # return preprocessed list of tokens
    return mytokens


# In[ ]:


# Custom transfer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        #Cleaning Text
        return [clean_text(text) for text in X]
    
    def fit(self, X, y=None, **fit-params):
        return self
    
    def get_params(self, deep=True):
        return {}
    
# Basic function to clean the text
def clean_text(text):
    # Removing spaces and coverting text into lowercase
    return text.strip().lower()


# In[ ]:


bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))


# In[ ]:


tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)


# In[ ]:


from sklearn.model_selection import train_test_split

X = df_amazon['verified reviews'] # the feature we want to analyze
ylabels = df_amazon['feedback'] # the labels, or answers, we want to test against

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)


# In[ ]:


# Logistic pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),("vectorizer", bow_vector), ("classifier", classifier)])

# model generation
pipe.fit(X_train,y_train)


# In[ ]:


from sklearn import metrics
# Predicting with a test dataset
predicted = pipe.predict(X_test)

# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))


# In[ ]:





# In[ ]:





# In[ ]:




