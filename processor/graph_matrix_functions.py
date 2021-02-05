# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 23:05:00 2019

@author: Stephan
"""

import pandas as pd
import numpy as np



def TrainIdf(Trainingset, stop_words='english'):
    '''
    ----------
    Parameters:
    
    Trainingset -- List of Strings, which will be analyzed for Keywords
    ----------
    
    Keyword Extraction with Sk Learn
    
    TfidfVectorizer will comput CountMatrix and TFidf score for a given Dataset
    The Fit Transform call compute both and gives back a Matrix with tdIf score of
    the used words.
    
    After that a for loop will filter the Keywords according to the condition,
    that the score is higher then 0.1
    '''

    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(stop_words=stop_words)

    x = tfidf.fit_transform(Trainingset)
    
    KeySet = []

    for i in range(0,x.shape[0]):

        df = pd.DataFrame(x[i].T.todense(), index=tfidf.get_feature_names(), columns=["tfidf"])
        df = df.sort_values(by=["tfidf"],ascending=False)
        df = df[df > 0.1].dropna(axis = 0, how = "all")

        returnString = ''

        for s in df.index.to_list():
            returnString = returnString +' '+ s

        KeySet.append(returnString)
    
    return KeySet



def Countmatrix(ListofStrings, pandas=True, matrix=False):

    '''
    Builds the Countmatrix and afterwards multiplicates it with its own
    transposed to get the occurences of Keywords with one another

    Args:
        ListofStrings ():  List of multipe Arrays of Strings. Should all be Keywords
        pandas (): bol - return a pandas dataframe (ram heavy) or return scipy sparse matrix
        matrix (): bol - return a pandas dataframe (ram heavy) or return scipy sparse matrix

    Returns:

    '''
    
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(lowercase=False)
    vector = cv.fit_transform(ListofStrings)
    
    #Following Line is used not for Pairs in sentences! It shows the Co-occurance of words
    #with another but withouht the Sentence context
    Keys=vector
    
    pdCv = pd.DataFrame(Keys.toarray(), columns =cv.get_feature_names())
    
    if pandas == True:
        return pdCv

    elif matrix == True:
        return [Keys,cv.get_feature_names()]



def getPairsPandas(M):

        ''' Get Pairs from a Pandas Dataframe. Only goes along one Half of the Matrix
        !!!!!!Funktioniert Nicht !!!! - Corrupted!!!!'''

        Pairs = []
        row = -1
        for startpoint in range(1,len(M)):
            row = row+1
            for col in range(startpoint, len(M)):

                if M.iloc[row,col] > 0:
                    Pair=[M.index[row], M.index[col]]
                    Pairs.append(Pair)

        return Pairs




def GetPairsNumpy(KeyM):
    '''
    Create Pairs vor Gephi and Networkanalysis from KeyM- Tuple which consist of (sparse.matrix of vectorizer,
    feature.names)

    Args:
        KeyM (): Tuple - (sparse.matrix, features.names) from vectorizer TF-IDF

    Returns: List of coocurrance pairs

    '''
    M = KeyM[0]
    KeyW = KeyM[1]

    #Iterate over all Text and build Pairs from Keyword
    Pairs = []

    for i in range(0, M.shape[0]):

        Target = np.nonzero(M[i] > 0)
        KeyWlist = Target[1]
        projekt = i

        for FirstEnt in KeyWlist:
            for SecEnt in KeyWlist:

                if FirstEnt != SecEnt:
                    Pair = [KeyW[FirstEnt], KeyW[SecEnt], projekt]
                    Pairs.append(Pair)

    return Pairs



def GetPairsWithWeight(KeyM):
    '''
    Create Pairs vor Gephi and Networkanalysis. Creates Dataframe with Souce, Target and Weight.
    Weight correspondes to the number of occurences, in which the Keyword occured allong the give corpus.
    M is the Countmatrix from Countvectorizer and KeyW is the name of Keywords, which has to be extracted
     from Countvetorizer.

    Args:
        KeyM (): Tuple - (sparse.matrix, features.names) from vectorizer TF-IDF

    Returns: pandas Dataframe with pairs of coocurrant features within the sparse.matrix

    '''

    #KeyM entspricht nicht der KeyM aus diesem Skript! sonder ist ein Pandas Dataframe, der
    # aus dem Countvektroizer gewonnen wird
    #!!!!!!!

    Source_All = []
    Target_All = []
    Weight_All = []

    for x in range(0, len(KeyM.columns)):

        TrueSeries = KeyM.iloc[:][x][KeyM.iloc[:][x]==1] # geht durch jede Spalte des CV durch und sucht welche Keywords vorkommen

        print('Zeile '+str(x) + ' von '+str(len(KeyM.columns)))

        for Source in TrueSeries.index:

            for Target in TrueSeries.index:

                if Source != Target:

                    counter_Similar = 0

                    for i in range(0, len(Source_All)):

                        currentPair = (Source, Target)
                        checkPair = (Source_All[i], Target_All[i])
                        checkPair2 = (Target_All[i], Source_All[i])

                        if currentPair == checkPair or currentPair == checkPair2:

                            counter_Similar = counter_Similar + 1
                            Weight_All[i] = Weight_All[i] + 1

                    if counter_Similar == 0:

                        Source_All.append(Source)
                        Target_All.append(Target)
                        Weight_All.append(1)

    df = pd.DataFrame([Source_All, Target_All, Weight_All], index=['Source', 'Target', 'Weight'])

    return df



def VerbinungsDataframeErzeugen(dataFrame):
    '''
     Erstellen des Dataframes mit Verbindungen zwischen den Einzelnen Stichworten in der Schlagwortdatei
     ÜBERGAGE: dataFrame muss ein Pandas Dataframe der eine Matrix enthält, die zu Verbindungen umgewandelt werden sollen
     Die Verbindungen werden entlang der Spalten erzeugt!!
     Bsp:
       a    b    c
     a1 1    2    0
     a2 0    3    0
     a3 1    0    1

     Wird zu [1,0,1],[2,3,0],[0,0,1]
     Verbindung wird Paarweise erzeugt aus den Listen. Also:
     Beispiel für [2,3,0]
     Source Targert
       2       3
       2       0
       3       2
       3       0
       0       2
       0       3

    Args:
        dataFrame ():

    Returns:

    '''

    gephiVerb = pd.DataFrame(columns=['Source', 'Target', 'Column'])


    # Geht jede Spalte/Porjekt ind DF durch
    for Zeilegesamt in range(0, len(dataFrame.columns)):

        zwischen = []

        # Hier werden die Nennungen eines Stichwortes, die in DF als Zahl hinterlegt sind (1 oder höher) zusammengesetzt als Liste
        # Für jedes Stichwort/Zeile in Stichwort
        for i in range(0, len(dataFrame.index)):

            # Wenn Stichwort mit 1 oder Höher gemarkt ist in der Matrix wird
            if dataFrame.iloc[i][Zeilegesamt] == 1:
                zwischen.append(i)

            elif dataFrame.iloc[i][Zeilegesamt] > 1:
                for ae in range(0, dataFrame.iloc[i][Zeilegesamt]):
                    zwischen.append(i)

        # Zwischen ist die List, welche erzeugt wird. Diese wird weitergegeben
        # Jeder Eintrag in Zwischen wird durchgegangen. Es werden in einem neuen Dataframe die Verbindungen in der Liste hinterlegt. Dafür werden die Datan als Paare zusammengefasst, bis jeder Eintrag aus der Liste als Paar vorhanden ist.
        for c in range(0, int(len(zwischen))):

            for f in range(0, int(len(zwischen))):

                int(zwischen[c]), int(zwischen[f])
                df3 = pd.DataFrame(index=range(0, 1), columns=['Source', 'Target', 'Column'])

                if not c == f:
                    df3.iloc[0][0] = zwischen[c]
                    df3.iloc[0][1] = zwischen[f]
                    df3.iloc[0][2] = dataFrame.columns[Zeilegesamt]

                    gephiVerb = pd.concat([df3, gephiVerb])

    gephiVerb.index = np.arange(len(gephiVerb))

    return (gephiVerb)


if __name__ == "__main__":
    
    '''Example how to use the Program'''
    from sklearn.datasets import fetch_20newsgroups
    categories = ['sci.med']     # Weitere Categorien ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

    #Downloads Testdataset from sklearn!
    twenty_train = fetch_20newsgroups(subset = 'train',
                                      categories=categories,
                                      remove=['footers', 'quotes'],
                                      shuffle = True,
                                      random_state = 42)

    # Funktion um Test zu ermöglichen
    KeyArray = TrainIdf(twenty_train.data)

    #Erzeugt Matrix für Keywordextraction
    KeyM = Countmatrix(KeyArray, pandas=False, matrix=True)

    #Erzeugt coocurance Pairs für spätere Netzwerk Analyse
    KPairs = GetPairsNumpy(KeyM)

    print('Corpus mit {count} Einträgen geladen'.format(count=len(KeyArray)))

    print('Matrix vom Type {typ} mit {cols} Spalten und {rows} Reihen erzeigt'.format(typ=KeyM[0].getformat(),
                                                                                      cols=KeyM[0].get_shape()[1],
                                                                                      rows=KeyM[0].get_shape()[0]),)
    print('Anzahl Erzeugte KPairs: %i' % (len(KPairs)))





