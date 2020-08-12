import pandas as pd
#from multi_rake import Rake
import Text_Class_Funktionen  as tcf


def getTextrankKeywords(string, stop_words=None):
    '''Gibt Keywords durch Textrank aus

    Aufruf z.B.

    from nltk.corpus import stopwords
    stop_words = stopwords.words("german")

    t=getTextrankKeywords('Blablalbalbl hallo', stop_words)
    print(t)

    '''
    if not stop_words:
        from nltk.corpus import stopwords

        stop_words = stopwords.words("german")

    tr4w = tcf.TextRank4Keyword()
    tr4w.analyze(string,
                 candidate_pos=['NOUN', 'PROPN'],
                 window_size=4,
                 lower=False,
                 stopwords=stop_words)
    keywords_TR = tr4w.get_keywords(number=10)
    return (keywords_TR)



def tfIdf_Keywords_train(corpus, stopword = None):


    if stopwords:
        stopWords = stopword
    else:
        from nltk.corpus import stopwords
        stop_words = stopwords.words("german")
        stop_words.extend(
            ['m²', 'FKZ', 'VE', '/', '„', 'KG', 'Co.', 'RAG', 'DMT', 'Ptj', 'K', 'PHI', 'u.a.', 'u.ä.', 'DUE',
             'B.&S.U.',
             'ITC', 'ISE', 'ERC', 'm', 'Ost', 'Süd', 'EA', 'EASD', 'A.3', 'B.3', 'LES', 'TILSE', 'IAP', 'LWS', 'HEM',
             '8)',
             'Test', 'FTA', 'FH', 'A1', 'ITI', 'Nord', 'West', 'Ruhr', 'OPVT', 'm2', 'HH', 'HTW', 'TT-WB/ENS1',
             'TT/MKX',
             'ILK', 'm³', 'Standort', 'Adlershof', 'EBC', 'EEC', 'OPV', 'IKT', 'GSG', 'Ort', 'PCS', 'Projektabschnitt',
             'kWh/m²', 'Ergebnisse', 'Voraussetzung', 'ScenoCalc', 'Schritt', 'Anbieter', 'Glaubwürdigkeit', 'Bezug',
             'Validierung', 'Gebiet', 'Verbundprojekt', 'Equipment', 'Fragestellungen', 'Einfluss', 'Auswahl',
             'Relation',
             'Indevo', 'Projektpartnern', 'Anzahl', 'Angebot', 'Bedarf'])

        stopWords = stop_words

    from sklearn.feature_extraction.text import CountVectorizer
    import re
    docs = corpus

    # create a vocabulary of words,
    # ignore words that appear in 85% of documents,
    # eliminate stop words
    cv = CountVectorizer(max_df=0.85, stop_words=stopWords)
    word_count_vector = cv.fit_transform(docs)

    # you only needs to do this once, this is a mapping of index to
    feature_names = cv.get_feature_names()

    from sklearn.feature_extraction.text import TfidfTransformer

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    return cv, tfidf_transformer



def tfIdf_Keywords_getKeyW(targetCorpus,cv,tfidf_transformer, nKey=10):


    '''From https://kavita-ganesan.com/extracting-keywords-from-text-tfidf/'''
    def sort_coo(coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def extract_topn_from_vector(feature_names, sorted_items, topn=10):
        """get the feature names and tf-idf score of top n items"""

        # use only topn items from vector
        sorted_items = sorted_items[:topn]

        score_vals = []
        feature_vals = []

        # word index and corresponding tf-idf score
        for idx, score in sorted_items:
            # keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])

        # create a tuples of feature,score
        # results = zip(feature_vals,score_vals)
        results = {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]] = score_vals[idx]

        return results


    # get the document that we want to extract keywords from
    doc = targetCorpus

    feature_names = cv.get_feature_names()

    # generate tf-idf for the given document
    tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    # extract only the top n; n here is 10
    keywords = extract_topn_from_vector(feature_names, sorted_items, nKey)

    # now print the results
    print("\n=====Doc=====")
    print(doc)
    print("\n===Keywords===")
    for k in keywords:
        print(k, keywords[k])





def tfIdf_Vectorizer_train(corpus,
                           standard=True, 
                           ngrams=False,
                           stop_word = None, 
                           ngram_range = (1,2)):
    '''Function to get TF-IDF Frequencey matrix from TfidfVectorzier'''

    if stop_word:

        stop_words = stop_word

    else:
        from nltk.corpus import stopwords
        stop_words = stopwords.words("german")
        stop_words.extend(
           ['m²', 'FKZ', 'VE', '/', '„', 'KG', 'Co.', 'RAG', 'DMT', 'Ptj', 'K', 'PHI', 'u.a.', 'u.ä.', 'DUE', 'B.&S.U.',
           'ITC', 'ISE', 'ERC', 'm', 'Ost', 'Süd', 'EA', 'EASD', 'A.3', 'B.3', 'LES', 'TILSE', 'IAP', 'LWS', 'HEM', '8)',
            'Test', 'FTA', 'FH', 'A1', 'ITI', 'Nord', 'West', 'Ruhr', 'OPVT', 'm2', 'HH', 'HTW', 'TT-WB/ENS1', 'TT/MKX',
            'ILK', 'm³', 'Standort', 'Adlershof', 'EBC', 'EEC', 'OPV', 'IKT', 'GSG', 'Ort', 'PCS', 'Projektabschnitt',
            'kWh/m²', 'Ergebnisse', 'Voraussetzung', 'ScenoCalc', 'Schritt', 'Anbieter', 'Glaubwürdigkeit', 'Bezug',
            'Validierung', 'Gebiet', 'Verbundprojekt', 'Equipment', 'Fragestellungen', 'Einfluss', 'Auswahl', 'Relation',
            'Indevo', 'Projektpartnern', 'Anzahl', 'Angebot', 'Bedarf'])

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

def tfIdf_Vectorizer_getKeyWords(df, column , n=10):
    '''Funktion um aus String Keywords zu extrahieren. Muss vorher aber den Vectorizer Trainieren'''
    return df.loc[column].sort_values(ascending=False)[:n]




def getRakeKeywords(string,stopwords1):
    '''Bringt nicht gewünschte Ergebnisse'''
    r = Rake(min_chars=3,
             max_words=2,
             min_freq=1,
             language_code='de',  # 'en'
             stopwords=None,  # {'and', 'of'}
             lang_detect_threshold=50,
             max_words_unknown_lang=2,
             generated_stopwords_percentile=80,
             generated_stopwords_max_len=3,
             generated_stopwords_min_freq=2
             )
    keywords_R = r.apply(string)
    return (keywords_R[:10])