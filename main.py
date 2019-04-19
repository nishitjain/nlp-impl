# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 22:41:49 2019
Description: Main file - The engine of the code.
@author: Nishit Jain
"""
import functions
import pandas as pd
from nltk.corpus import stopwords
from string import punctuation
from sklearn.naive_bayes import MultinomialNB
import traceback
import warnings
import logging as log

warnings.filterwarnings('ignore')
log.basicConfig(filename='messages.log',
                    level=log.DEBUG,
                    format='%(asctime)s - %(name)s - %(threadName)s -  %(levelname)s - %(message)s')


if __name__=='__main__':
    try:
        # Getting custom stop words list & train and test dataset
        custom_stopwords_list = list(set(stopwords.words('english')))+list(punctuation)+['``', "'s", "...", "n't","-","_",":","&","<",">","--","xxxx","xx/xx/xxxx"]
        df = pd.read_pickle('Consumer_complaints.pkl')
        log.info('Shape of Data Frame: {}'.format(df.shape))
        df = df[['Consumer complaint narrative','Product']]
        df.dropna(inplace=True)
        
        # Getting distinct category names.
        category_names = df.Product.value_counts().index.tolist()
        log.info('Distinct Category Names: {}'.format(category_names))
        
        # Performing Text Classification Using Count & TF-IDF Vectorizer
        count_acc, count_report = functions.text_classifier_(df['Consumer complaint narrative'],df.Product,category_names,MultinomialNB(),custom_stopwords_list,vectorizer_type='count')
        tfidf_acc, tfidf_report = functions.text_classifier_(df['Consumer complaint narrative'],df.Product,category_names,MultinomialNB(),custom_stopwords_list,vectorizer_type='tfidf')
        log.info('Count Vectorizer Accuracy: {}'.format(count_acc))
        log.info('Count Vectorizer Classification Report: \n{}'.format(count_report))
        log.info('TF-IDF Accuracy: {}'.format(tfidf_acc))
        log.info('TF-IDF Classification Report: \n{}'.format(tfidf_report))
        
        # Topic Modelling (LSI - SVD)
        fina_topic_list = functions.lsi_model_svd_(df['Consumer complaint narrative'],custom_stopwords_list,num_topics=10,num_words=7)
        log.info('Final Topic List: \n{}'.format(fina_topic_list))
        
        # Topic Modelling using LDA.
        ldamodel, coherence_score_lda, perplexity_score_lda = functions.lda_model_(df['Consumer complaint narrative'], custom_stopwords_list, num_topics=10)
        log.info('LDA Topics: \n{}'.format(ldamodel.print_topics()))
    except:
        print('An exception Occurred. {}'.format(traceback.format_exc()))
        log.error(traceback.format_exc())