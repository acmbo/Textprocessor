# -*- coding: utf-8 -*-
"""
Created on Fri Nov  12 12:03:00 2019

@author: Stephan
"""

from processor.Textprocessor import Textprocessor

if __name__=='__main__':

    from nltk.corpus import stopwords

    stop_words = stopwords.words("german")  # orginal ist in "english",


    text = 'Zelda ist cool. Die Yuhan und Mario hat das neue Zelda. Shutong aber noch nicht.'
    text2 ='Hallo Welt. Oder lieber Hello World! Oder Konichiwa! Sumimasem ! Arigatoooo! Mario ist doof!'

    cp = Textprocessor([text, text2], stopwords=stop_words)  # Für eigene Stopwörter hier stopwords = stop_words übergeben

    #cp.preprossText()

    kw = cp.ExtractKeywords()

    cp.CreatePairsFromKeywords()  # Falls Edges für Gephi abgespeichert werden sollen saveFile = True übergeben

    g = cp.CreateNetwork()

    print(kw)

    from pyvis.network import Network

    g_in=Network()
    g_in.toggle_hide_edges_on_drag(True)
    g_in.barnes_hut()
    g_in.from_nx(g)
    g_in.show_buttons()
    g_in.show('text.html')