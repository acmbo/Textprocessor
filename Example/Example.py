# -*- coding: utf-8 -*-
"""
Created on Fri Nov  12 12:03:00 2019

@author: Stephan
"""
import json
import context
from processor.Textprocessor import Textprocessor

if __name__=='__main__':

    # collect stopwords
    from nltk.corpus import stopwords
    stop_words = stopwords.words("english")

    # Import spacy and language model, necessary for textRank algorithm
    import spacy
    nlp = spacy.load("en_core_web_md")

    # Open Exmaple file with corpus
    with open('example_data.txt') as json_file:
        data = json.load(json_file)

    corpus = list(data.values())

    cp = Textprocessor(corpus ,nlp, stopwords=stop_words)

    #cp.preprossText()

    kw = cp.ExtractKeywords(Algo='Textrank',n=20)
    print(kw)

    kw = cp.ExtractKeywords(Algo='TFIDF', n=20)
    print(kw)

    cp.CreatePairsFromKeywords()  # Falls Edges für Gephi abgespeichert werden sollen saveFile = True übergeben

    g = cp.CreateNetwork()


    # Vizualization through pyviz. please decomment the following lines. This will create an html file with the created
    # network.

    #from pyvis.network import Network
    #g_in=Network()
    #g_in.toggle_hide_edges_on_drag(True)
    #g_in.barnes_hut()
    #g_in.from_nx(g)
    #g_in.show_buttons()
    #g_in.show('text.html')