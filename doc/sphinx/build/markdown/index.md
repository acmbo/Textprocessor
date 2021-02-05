
# Welcome to Textprocessor’s documentation!

# TeacherAPI models

Created on Thur April 11 19:25:22 2019

@author: Stephan


### processor.extractionAlgorithms.getRakeKeywords(string, stopwords1)
Bringt nicht gewünschte Ergebnisse


### processor.extractionAlgorithms.getTextrankKeywords(string, stop_words=None)
Gibt Keywords durch Textrank aus

Aufruf z.B.

from nltk.corpus import stopwords
stop_words = stopwords.words(„german“)

t=getTextrankKeywords(‚Blablalbalbl hallo‘, stop_words)
print(t)


### processor.extractionAlgorithms.tfIdf_Keywords_getKeyW(targetCorpus, cv, tfidf_transformer, nKey=10)
From [https://kavita-ganesan.com/extracting-keywords-from-text-tfidf/](https://kavita-ganesan.com/extracting-keywords-from-text-tfidf/)


### processor.extractionAlgorithms.tfIdf_Vectorizer_getKeyWords(df, column, n=10)
Funktion um aus String Keywords zu extrahieren. Muss vorher aber den Vectorizer Trainieren


### processor.extractionAlgorithms.tfIdf_Vectorizer_train(corpus, standard=True, ngrams=False, stop_word=None, ngram_range=(1, 2))
Function to get TF-IDF Frequencey matrix from TfidfVectorzier

# Indices and tables


* Stichwortverzeichnis


* Modulindex


* Suche
