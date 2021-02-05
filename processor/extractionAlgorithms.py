# -*- coding: utf-8 -*-
"""
Created on Thur April 11 19:25:22 2019

@author: Stephan
"""


import numpy as np
import pandas as pd
from processor import TextRank as tR


def getTextrankKeywords(string, nlp, stop_words=None, n=10):
    '''
    Creates keywords through TextRank algorithm.

    Args:
        string (): string - String of a text, in which keywords are to find in
        stop_words (): list of stopwords
        n ():  int - amount of keywords

    Returns: Keywords as tuple

    '''
    if not stop_words:
        from nltk.corpus import stopwords

        stop_words = stopwords.words("german")

    tr4w = tR.TextRank4Keyword(nlp=nlp)
    tr4w.analyze(string,
                 candidate_pos=['NOUN', 'PROPN'],
                 window_size=4,
                 lower=False,
                 stopwords=stop_words)
    keywords_TR = tr4w.get_keywords(number=n)
    return (keywords_TR)



def tfIdf_Keywords_train(corpus, stopword = None):
    '''
    Train vectorizer and returns the trained count vectorizer

    Args:
        corpus (): list of strings of texts, where algorithm needs to find keywords from
        stopword (): list of stopwords

    Returns:

    '''

    if stopword:
        stop_Words = stopword
    else:
        from nltk.corpus import stopwords
        stop_Words = stopwords.words("german")


    from sklearn.feature_extraction.text import CountVectorizer

    docs = corpus

    # create a vocabulary of words,
    # ignore words that appear in 85% of documents,
    # eliminate stop words
    cv = CountVectorizer(max_df=0.85, stop_words=stop_Words)
    word_count_vector = cv.fit_transform(docs)

    # you only needs to do this once, this is a mapping of index to
    # feature_names = cv.get_feature_names()

    from sklearn.feature_extraction.text import TfidfTransformer

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    return cv, tfidf_transformer




def tfIdf_Vectorizer_train(corpus,
                           standard=True, 
                           ngrams=False,
                           stop_word = None, 
                           ngram_range = (1,2)):
    '''
    Function to get TF-IDF Frequencey matrix from TfidfVectorzier

    Args:
        corpus (): list of strings of texts, where algorithm needs to find keywords from
        standard (): bol - returns a pands dataframe if true (ram heavy) or returns sparse.matrix if false
        ngrams (): bol - if True, tf-IDF will calculate n-grams
        stop_word (): list of stopwords
        ngram_range (): tuple - if ngrams==True here you can decide what kind of ngrams should be used.
                                See tf_idf documention in sklearn for mor information

    Returns: tf-IDF vector either as pandas DF or as sparse matrix

    '''

    if stop_word:
        stop_words = stop_word
    else:
        from nltk.corpus import stopwords
        stop_words = stopwords.words("german")


    from sklearn.feature_extraction.text import TfidfVectorizer

    if ngrams:
        tfidf_vectorizer = TfidfVectorizer(lowercase=False,stop_words=stop_words,use_idf=True,ngram_range=ngram_range)
    else:
        tfidf_vectorizer = TfidfVectorizer(lowercase=False, stop_words=stop_words, use_idf=True)

    # just send in all your docs here
    fitted_vectorizer = tfidf_vectorizer.fit(corpus)
    tfidf_vectorizer_vectors = fitted_vectorizer.transform(corpus)


    if standard:

        names = fitted_vectorizer.get_feature_names()
        data = tfidf_vectorizer_vectors.todense().tolist()
        df = pd.DataFrame(data, columns=names)
    
        # remove all columns containing a stop word from the resultant dataframe.
    
        for col in df.columns:
            if col in stop_words:
                df = df.drop([col], axis=1)
    
        tfIdf_df = df
        return tfIdf_df
    
    else:
        
        names = fitted_vectorizer.get_feature_names()
        return(names,tfidf_vectorizer_vectors)



def tfIdf_Vectorizer_getKeyWords(df, column , n=10, intype='pandas'):
    '''
    Function to return keywords from a tf_idf vectorizer in pandas
    Args:
        df (): Dataframe of vectors
        column (): Target Keyword
        n (): number of keywords

    Returns:

    '''
    if intype=='pandas':
        return df.loc[column].sort_values(ascending=False)[:n]
    elif intype=='sparse':
        return list(reversed([df[0][x] for x in np.argsort(df[1][column].toarray())[0][-n:]]))
    else:
        return 0



def getRakeKeywords(string,stopwords1):
    '''Bringt nicht gewünschte Ergebnisse'''
    #from multi_rake import Rake
    #r = Rake(min_chars=3,
    #         max_words=2,
    #         min_freq=1,
    #         language_code='de',  # 'en'
    #         stopwords=None,  # {'and', 'of'}
    #         lang_detect_threshold=50,
    #         max_words_unknown_lang=2,
    #         generated_stopwords_percentile=80,
    #         generated_stopwords_max_len=3,
    #         generated_stopwords_min_freq=2
    #         )
    #keywords_R = r.apply(string)
    #return (keywords_R[:10])
    return 0