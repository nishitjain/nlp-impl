# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 22:41:03 2019
@author: Nishit Jain
Description: Functions to implement various kinds of NLP techniques using customer 
            complaints dataset.
"""

from nltk.stem.wordnet import WordNetLemmatizer
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LsiModel
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
import logging as log

warnings.filterwarnings('ignore')
log.basicConfig(filename='messages.log',
                    level=log.DEBUG,
                    format='%(asctime)s - %(name)s - %(threadName)s -  %(levelname)s - %(message)s')

def preprocess_(document,custom_stopwords_list):
    log.info('Into the preprocessing function.')
    stop_words_removed = " ".join([token for token in document.lower().split() if token not in custom_stopwords_list])
    lemmatized = " ".join([WordNetLemmatizer().lemmatize(token) for token in stop_words_removed.split()])
    log.info('Tokenization & Lemmatization completed.')
    return lemmatized

def run_model_(model,x,y):
    log.info('Splitting Data.')
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
    log.info('Running model & making predictions.')
    y_pred = model.fit(x_train,y_train).predict(x_test)
    log.debug('Accuracy: {} \n Calssification Report:\n {}'.format(accuracy_score(y_test,y_pred),classification_report(y_test,y_pred)))
    return accuracy_score(y_test,y_pred),classification_report(y_test,y_pred)

def label_encode_(x, category_names):
    for e in category_names:
        if x == e:
            return category_names.index(e)

def vectorizer_(text, custom_stopwords_list, vectorizer='count'):
    if(vectorizer=='count'):
        log.info('In Count Vectorizer.')
        model_vectorizer = CountVectorizer(stop_words=custom_stopwords_list)
    if(vectorizer=='tfidf'):
        log.info('In TF-IDF Vectorizer.')
        model_vectorizer = TfidfVectorizer(stop_words=custom_stopwords_list)
    x = model_vectorizer.fit_transform(text)
    #log.debug('Length of vector: {}'.format(len(x)))
    return x

def text_classifier_(X, y, category_names, model, custom_stopwords_list, vectorizer_type='count'):
    log.info('In Text Classifier function.')
    y = y.apply(lambda x : label_encode_(x, category_names))
    if(vectorizer_type=='count'):
        return run_model_(model, vectorizer_(X, custom_stopwords_list), y)
    if(vectorizer_type=='tfidf'):
        return run_model_(model, vectorizer_(X, custom_stopwords_list, vectorizer='tfidf'), y)

def lsi_model_svd_(X, custom_stopwords_list, num_topics, num_words):
    log.info('In LSI Model with SVD function.')
    num_words += 1
    tfidf = TfidfVectorizer(stop_words=custom_stopwords_list)
    vector = tfidf.fit_transform(X.str.lower())
    #X = pd.DataFrame(vector)
    model = TruncatedSVD(n_components=num_topics, algorithm='randomized', n_iter=100, random_state=122)
    model.fit(vector)
    terms = tfidf.get_feature_names()
    topics = []
    for i, comp in enumerate(model.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
        topics.append("Topic "+str(i)+": ")
        for t in sorted_terms:
            topics.append(t[0])
    final_topic_list = [topics[i:i+num_words] for i in range(0, len(topics), num_words)]
    log.debug('Final Topics List: {}'.format(final_topic_list))
    return final_topic_list
        
def lda_model_(documents, custom_stopwords_list, num_topics):
    log.info('In LDA for Topic Modelling function.')
    cleaned_documents = [preprocess_(document, custom_stopwords_list).split() for document in documents]
    document_dictionary = corpora.Dictionary(cleaned_documents)
    td_matrix = [document_dictionary.doc2bow(document) for document in cleaned_documents]
    model =LdaModel(corpus=td_matrix,num_topics=num_topics,id2word=document_dictionary,passes=100)
    coherence_model = CoherenceModel(model=model,texts=cleaned_documents,dictionary=document_dictionary,coherence='c_v')
    log.debug('Coherence Score: {}\nPerplexity Score: {}'.format(coherence_model.get_coherence(), model.log_perplexity(td_matrix)))
    return model, coherence_model.get_coherence(), model.log_perplexity(td_matrix)

def lsi_model_(documents, custom_stopwords_list, num_topics):
    log.info('In LSI using gensim for Topic Modelling function.')
    cleaned_documents = [preprocess_(document, custom_stopwords_list).split() for document in documents]
    document_dictionary = corpora.Dictionary(cleaned_documents)
    td_matrix = [document_dictionary.doc2bow(document) for document in cleaned_documents]
    model = LsiModel(corpus=td_matrix,id2word=document_dictionary, num_topics=num_topics)
    coherence_model = CoherenceModel(model=model,texts=cleaned_documents,dictionary=document_dictionary,coherence='c_v')
    log.debug('Coherence Score: {}\nProjection Score (Single Values per topic): {}'.format(coherence_model.get_coherence(), model.projection.s))
    return model, coherence_model.get_coherence(), model.projection.s

