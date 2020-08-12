# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 23:05:00 2019

@author: Stephan
"""

import pandas as pd
import numpy as np
#from graph_tool.all import *


def TrainIdf(Trainingset):
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
    tfidf = TfidfVectorizer(stop_words='english')
    x = tfidf.fit_transform(twenty_train.data)
    
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
    '''Builds the Countmatrix and afterwards multiplicates it with its own 
    transposed to get the occurences of Keywords with one another
    
    Parameters:
        ListofStrings -- List of multipe Arrays of Strings. Should all be Keywords
    '''
    
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.corpus import stopwords

    stop_words2 = stopwords.words("german")
    cv = CountVectorizer(lowercase=False,stop_words=stop_words2)
    vector = cv.fit_transform(ListofStrings)
    
    #Following Line is used not for Pairs in sentences! It shows the Co-occurance of words
    #with another but withouht the Sentence context
    #Keys= vector.transpose().dot(vector)
    Keys=vector
    
    pdCv = pd.DataFrame(Keys.toarray(), columns =cv.get_feature_names())
    
    if pandas == True:
        return pdCv
    elif matrix == True:
        return [Keys,cv.get_feature_names()]



def getPairsPandas(M):
        ''' Get Pairs from a Pandas Dataframe. Only goes along one Half of the Matrix
        !!!!!!Funktioniert Nicht !!!!'''
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
    '''Create Pairs vor Gephi and Networkanalysis'''
    M = KeyM[0]
    KeyW = KeyM[1]

    ''' Iterate over all Text and build Pairs from Keyword'''
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
    '''Create Pairs vor Gephi and Networkanalysis. Creates Dataframe with Souce, Target and Weight. Weight correspondes
    to the number of occurences, in which the Keyword occured allong the give corpus. M is the Countmatrix from Countvectorizer
    and KeyW is the name of Keywords, which has to be extracted from Countvetorizer.'''

    #KeyM entspricht nicht der KeyM aus diesem Skript! sonder ist ein Pandas Dataframe, der aus dem Coountvektroizer gewonnen wird
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





    for i in range(0, M.shape[0]):
        Target = np.nonzero(M[0] > 0)
        KeyWlist = Target[1]

        for FirstEnt in KeyWlist:
            for SecEnt in KeyWlist:

                if FirstEnt != SecEnt:
                    Pair = [KeyW[FirstEnt], KeyW[SecEnt]]
                    Pairs.append(Pair)
    return Pairs


def VerbinungsDataframeErzeugen(dataFrame):
    # Erstellen des Dataframes mit Verbindungen zwischen den Einzelnen Stichworten in der Schlagwortdatei
    # ÜBERGAGE: dataFrame muss ein Pandas Dataframe der eine Matrix enthält, die zu Verbindungen umgewandelt werden sollen
    # Die Verbindungen werden entlang der Spalten erzeugt!!
    # Bsp:
    #   a    b    c
    # a1 1    2    0
    # a2 0    3    0
    # a3 1    0    1

    # Wird zu [1,0,1],[2,3,0],[0,0,1]
    # Verbindung wird Paarweise erzeugt aus den Listen. Also:
    # Beispiel für [2,3,0]
    # Source Targert
    #   2       3
    #   2       0
    #   3       2
    #   3       0
    #   0       2
    #   0       3

    # Die Ausgabe sollte immer in einen neuen Dataframe gespeicher werden mit
    # df_neu = VerbinungsDataframeErzeugen(df).copy()

    # d=0
    gephiVerb = pd.DataFrame(columns=['Source', 'Target', 'Column'])
    # zwischen2=[]
    # Zeilegesamt=147
    # Geht jede Spalte/Porjekt ind DF durch
    for Zeilegesamt in range(0, len(dataFrame.columns)):
        zwischen = []
        # Hier werden die Nennungen eines Stichwortes, die in DF als Zahl hinterlegt sind (1 oder höher) zusammengesetzt als Liste
        # Für jedes Stichwort/Zeile in Stichwort
        for i in range(0, len(dataFrame.index)):
            # Wenn Stichwort mit 1 oder Höher gemarkt ist in der Matrix wird
            if dataFrame.iloc[i][Zeilegesamt] == 1:
                zwischen.append(i)
                # print(zwischen)
            elif dataFrame.iloc[i][Zeilegesamt] > 1:
                for ae in range(0, dataFrame.iloc[i][Zeilegesamt]):
                    # print(df.iloc[i][Zeilegesamt])
                    zwischen.append(i)
                    # print(zwischen)
                    # print (ae)
        print(zwischen)

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
                    # print (df3)
                    gephiVerb = pd.concat([df3, gephiVerb])

    gephiVerb.index = np.arange(len(gephiVerb))
    return (gephiVerb)


if __name__ == "__main__":
    
    '''Example how to use the Program'''
    from sklearn.datasets import fetch_20newsgroups
    categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
    
    twenty_train = fetch_20newsgroups(subset = 'train', 
                                      categories=categories, 
                                      shuffle = True,
                                      random_state = 42)
    
    KeyArray = TrainIdf(twenty_train.data)  
    KeyM = Countmatrix(KeyArray, pandas=False, matrix=True)
    KPairs = GetPairsNumpy(KeyM)




