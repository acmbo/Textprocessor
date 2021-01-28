# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:32:17 2019

@author: Stephan
"""

from processor import TextRank as tcf
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from germalemma import GermaLemma

#Further Pacakges for Preprocessing https://github.com/jfilter/german-preprocessing
#Germanlemma: https://github.com/WZBSocialScienceCenter/germalemma

#------------------------------Paramterter und Bezeichnung---------------------
gerLem = GermaLemma()
lem = WordNetLemmatizer()
stem = PorterStemmer()
#---------------------------------Funktionen-------------------------------------------


def PrePross(ListofSentences,_comma=False, Fuzzy=False, FuzzyRank=False,
             _reversed = False, Remove_specCar = False, IgnoreWord_list = [None],
             stem=False, stopwords=[]):
    '''
    Funktion um den Text vorbereiten. Braucht einen Dataframe und den Columnnamen,
    #indem sich die texte befinden. Im

    Args:
        ListofSentences (): Liste mit Textdaten
        _comma (): Bol - soll
        Fuzzy ():  Aktiviert fuzzy ersetzung, dabei wird abgespeichert, welche Wörter bereits vorkamen und diese Wörter bzw Ähnliche Wörter zu diesen werden nicht mehr ersetzt
        FuzzyRank (): Fuzzy Search, der die Wörter nach ihrem erscheinen ranked und dann die Fuzzy search darüber laufen lässt
        _reversed ():  Lässt Fuzzy Search ablaufen nur start vom Ende der luste und nicht vom Anfang an
        Remove_specCar (): Entfernt sepcial Charakters in einem extra schritt zusätzlich falls erster schritt nicht genügt, führt leider momentan auch zur entfernung des Punktes am Ende des Satzes
        IgnoreWord_list (): Wörter die durch Fuzzy Search ignoriert werden sollten

    Returns:

    '''

    #Entfernen der Stoppwörter
    #TextProxessing aller BEschreibungen. Satzpunkte werden hierbei gelöscht!
    #Es wird nur ein gemeinsamer String ohne erkennung wo ein satz aufhört gebildet

    #Benötigte Parameter für Fuzzy
    fuzzy_untereGrenze = 96
    fuzzy_obereGrenze = 100
    fuzzy_candidatenuntereSchwelle = 96



    import spacy
    from nltk import word_tokenize
    nlp = spacy.load("de_core_news_lg")



    def applyregextoCopurs(KB, func, exceptionlist=[]):
        '''
        Function for seraching a corpus with a given matching function,
        which targets strings. It erases Entry which are found by the function
        KB - original Corpus
        func - (str) regexfunction to apply e.g re.findall("^([0-9]{2}:?)([A-Z])([A-Z])([0-9])", w)

        Example:  fkz, ListofSentences = applyregextoCopurs(ListofSentences, "^([0-9]{2}:?)([A-Z])([A-Z])([0-9])")
        '''

        finds = []
        corpus = []
        for kb in KB:
            words = word_tokenize(kb)
            new_words = [w for w in words if not re.findall(func, w) or w in exceptionlist]
            finds.extend([w for w in words if re.findall(func, w) and w not in exceptionlist])
            corpus.append(' '.join(new_words))
        return finds, corpus


    return_list = []

    for x in range(0,len(ListofSentences)):

        #Recreates Bindwort with '-' inbetween as one Word without '-'

        tokens = word_tokenize(ListofSentences[x])
        bow = []
        for w in tokens:
            w = w.replace('-', ' ')
            w = w.replace('\n',' ')
            bow.append(w.lower())
        text = " ".join(bow)

        ##Remove punctuations
        text = re.sub('[^a-zA-ZäüöÄÜÖß.]', ' ', text)
        
        #Convert to lowercase
        text = text.lower()
        
        #remove tags
        text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
        
        # remove special characters and digits
        #text=re.sub("(\\d|\\W)+"," ",text) # Führt Dazu das Punkt als Satzende entfällt. Ist problematisch für Textrank


        if Remove_specCar:
            text=tcf.entferneStopper(text) # Führt zu Entfernung von Punkt am ende des Satzes
            
        else:
             text = word_tokenize(text)
             text = ",".join(text)
        
        ##Convert to list from string
        text = text.split(',')

            
        if len(stopwords)>0:
            text = [word for word in text if word not in stopwords]
            

        ##Stemming und Lemmatisation
        if stem:
            
            def genlemma(text1):
                
                for word in text1:
                    
                    spacy_doc = nlp(word)
                    '''
                    if spacy_doc[0].pos_ == 'VERB':
                        r_value = gerLem.find_lemma(word,'VERB')
                        yield r_value 
                    elif spacy_doc[0].pos_== 'NOUN':
                        r_value = gerLem.find_lemma(word,'N')
                        yield r_value 
                    elif spacy_doc[0].pos_ =='ADJ':
                        r_value = gerLem.find_lemma(word,'ADJ')
                        yield r_value 
                    elif spacy_doc[0].pos_ =='ADV':
                        r_value = gerLem.find_lemma(word,'ADV')
                        yield r_value 
                    else:
                        r_value = gerLem.find_lemma(word,'N')
                        yield r_value
                    '''
                    r_value = gerLem.find_lemma(word,'N')
                    yield r_value
                              
                    
            #text = [gerLem.find_lemma(word,'N') for word in text] # Doppelter Stemmer und Lem
            gen1 = genlemma(text)
            text = list(gen1)

        spacy_doc = nlp(' '.join(text))  #Doppeler Stemmer und Lem
        
        #stemmed_tokens = [token.lemma_ for token in spacy_doc]
        stemmed_tokens = [token.text for token in spacy_doc]


        #Erzeugen eines neuen Corpuses
        if _comma:
            text = stemmed_tokens
        else:
            text = " ".join(stemmed_tokens)
            
        #Capitalize
        text = list(map(lambda x: x.capitalize() , text))
        text = " ".join(stemmed_tokens)

        return_list.append(text)



    if Fuzzy:

        from fuzzywuzzy import fuzz

        fuzzy_list = return_list

        tokenlists = []

        # Jeder Eintrag aus dem Corpus aus wird als list aus Tokenized Words in tokenlists gespeichert.
        for val in fuzzy_list:
            tok = word_tokenize(val)
            tokenlists.append(tok)

        candidates = [] # Kandidaten, mit denen Fuzzy search durchgeführt wurde


        for i in range(0,len(tokenlists)):

            for j,word in enumerate(tokenlists[i]):

                if word not in candidates:

                    for n in range(0,len(tokenlists)):

                        for m in range(0,len(tokenlists[n])):

                            Ratio = fuzz.ratio(word.lower(), tokenlists[n][m].lower())

                            if fuzzy_obereGrenze > Ratio > fuzzy_untereGrenze and word not in IgnoreWord_list:
                                print(str(tokenlists[n][m]) + ' durch ' + str(word))

                                file1 = open("KeywordsOutput/FuzzyOutput.txt", "a")
                                file1.write(str(tokenlists[n][m]) + ' durch ' + str(word) + ' wegen Ratio: '+ str(Ratio) +'; \n')
                                file1.close()

                                tokenlists[n][m] = word
                                if tokenlists[n][m] not in candidates:  # Dazugekommen
                                    candidates.append(tokenlists[n][m]) # Dazugekommen


                    candidates.append(word)

        return_list = tokenlists




    if FuzzyRank:

        from fuzzywuzzy import fuzz
        import collections

        fuzzy_list = return_list
        ranking_str = ' '.join(return_list) # String mit allen Wörtern die vorkommen

        tokenlists = []

        # Jeder Eintrag aus dem Corpus aus wird als list aus Tokenized Words in tokenlists gespeichert.
        for val in fuzzy_list:

            tok = word_tokenize(val)
            tokenlists.append(tok)

        candidates = [] # Kandidaten, mit denen Fuzzy search durchgeführt wurde

        #Zählen der Wörter
        wordcount = {}

        for word in ranking_str.split():

            if word not in wordcount:
                wordcount[word] = 1
            else:
                wordcount[word] += 1

        word_counter = collections.Counter(wordcount) # Counter Klasse um Wörter zu zählen. verwende: .most_common()


        commonWords = word_counter.most_common()

        if _reversed:
            commonWords = list(reversed(word_counter.most_common())) # Invertieren der liste


        for word, counter in commonWords:


            if counter > 1: # Verhindern das zu unbedeutenden Worte analysiert werden

                if len(candidates) ==0:

                    for n in range(0, len(tokenlists)):

                        for m in range(0, len(tokenlists[n])):

                            Ratio = fuzz.ratio(word.lower(), tokenlists[n][m].lower())

                            if fuzzy_obereGrenze > Ratio > fuzzy_untereGrenze and word not in IgnoreWord_list:
                                #print(str(word) + ':' + str(counter))
                                print(str(tokenlists[n][m]) + ' durch ' + str(word))

                                file1 = open("KeywordsOutput\\FuzzyOutput.txt", "a")
                                file1.write(str(tokenlists[n][m]) + ' durch ' + str(word) +'; \n')
                                file1.close()

                                tokenlists[n][m] = word
                                if tokenlists[n][m] not in candidates:
                                    candidates.append(tokenlists[n][m])

                    candidates.append(word)

                Temp = []

                for entry in candidates:
                    #Alle Entry Ratings berechen
                    EntryRatio = fuzz.ratio(word.lower(), entry.lower())

                    Temp.append(EntryRatio)

                Temp = np.asarray(Temp) # Array aus EntryRatios


                if not np.any(Temp > fuzzy_candidatenuntereSchwelle):
                    # Prüfen ob entry in Candidatenliste mit bestimmen Ration vorhanden ist


                    for n in range(0,len(tokenlists)):

                        for m in range(0,len(tokenlists[n])):

                            Ratio = fuzz.ratio(word.lower(), tokenlists[n][m].lower())

                            if fuzzy_obereGrenze > Ratio > fuzzy_untereGrenze and word not in IgnoreWord_list:
                                print(str(tokenlists[n][m]) + ' durch ' + str(word) + ' wegen Ratio: '+ str(Ratio))

                                file1 = open("KeywordsOutput\\FuzzyOutput.txt", "a")
                                file1.write(str(tokenlists[n][m]) + ' durch ' + str(word) + ' wegen Ratio: '+ str(Ratio) +'; \n')
                                file1.close()

                                tokenlists[n][m] = word

                                if tokenlists[n][m] not in candidates:
                                    candidates.append(tokenlists[n][m])

                    candidates.append(word)

        return_list = []
        for val in tokenlists:
            Wordlist = ' '.join(val)
            return_list.append(Wordlist)

    return(return_list)





def dfToExcel(dataframe, string):
    name=string+'.xlsx'
    writer = pd.ExcelWriter(name)
    dataframe.to_excel(writer,'Sheet1')
    writer.save()





# --------------Stemmmer--------------------
def entferneStopper(liste, stop_words=None):

    from nltk.tokenize import word_tokenize
    tokenized_word = word_tokenize(liste)

    if not stop_words:
        from nltk.corpus import stopwords
        stop_words = stopwords.words("german")  # orginal ist in "english", kann aber auch mit Deutsch

    filtered_sent = ''
    first = True

    for w in tokenized_word:

        # Entferne Klammern und andere
        chars = {'Ã¶': 'ö', 'Ã¤': 'ä', 'Ã¼': 'ü', 'ÃŸ': 'ß', ';': ' ', '(': ' ', ')': ' '}
        for char in chars:
            w = w.replace(char, chars[char])

        # Stemmer
        # stemmed_words.append(sno.stem(w))
        if w not in stop_words and first == False:
            filtered_sent = filtered_sent + ',' + str(w)

        if w not in stop_words and first == True:
            filtered_sent = filtered_sent + str(w)
            first = False

    return filtered_sent


# -------------------------------------------------------------------

def MeistverwendeteWorte(string):

    token = string.split()
    token2 = pd.Series(token)
    returnX = token2.value_counts()

    return (returnX)




