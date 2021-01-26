# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:32:17 2019

@author: jri-swe
"""


import Text_Class_Funktionen  as tcf
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
#from multi_rake import Rake
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from germalemma import GermaLemma
#https://datascience.blog.wzb.eu/2016/07/13/accurate-part-of-speech-tagging-of-german-texts-with-nltk/
from nltk.tokenize import word_tokenize

#Further Pacakges for Preprocessing https://github.com/jfilter/german-preprocessing
#Germanlemma: https://github.com/WZBSocialScienceCenter/germalemma


#------------------------------Paramterter und Bezeichnung---------------------
gerLem = GermaLemma()
lem = WordNetLemmatizer()
stem = PorterStemmer()
stop_words = stopwords.words("german")
#tr4w = tcf.TextRank4Keyword()

#Zusatzstopwords
stop_words.extend(['m²','FKZ','VE','/','„','KG','Co.','RAG','DMT','Ptj','K','PHI','u.a.','u.ä.','DUE','B.&S.U.',
                   'ITC','ISE','ERC','m','Ost','Süd','EA','EASD','A.3','B.3','LES','TILSE','IAP','LWS','HEM','8)',
                   'Test','FTA','FH','A1','ITI','Nord','West','Ruhr','OPVT','m2','HH','HTW','TT-WB/ENS1','TT/MKX',
                   'ILK','m³','Standort','Adlershof','EBC','EEC','OPV','IKT','GSG','Ort','PCS','Projektabschnitt',
                   'kWh/m²','Ergebnisse','Voraussetzung','ScenoCalc','Schritt','Anbieter','Glaubwürdigkeit','Bezug',
                   'Validierung','Gebiet','Verbundprojekt','Equipment','Fragestellungen','Einfluss','Auswahl',
                   'Relation','Indevo','Projektpartnern','Anzahl','Angebot','Bedarf', 'Ch','Dec','Dfki',
                   'Ee','Ens','Et','Fa','lt','Ksb','Kw','Nvz','Of','Shc','Sol','Swt','Ts','Tum','Tud','Tm', 'Ttwb','Tvb',
                   'Uvb','Ap','Arbeitspaket', 'Frauenhofer','Firma','Konzept','Covestro','Fresnel'])






#Öffnen der Stopwörter
with open("Quelldaten/TR_StopworteZusatz2.txt", "r",encoding='cp1252') as input:
#with open("Schlagworte_Neu_28_03_2019_OhneFKZ2.txt", "r") as input:
    externStop = input.read().split("\n")   #\n\n denotes there is a blank line in between paragraphs.


#Entfernen der Duplicate in liste
externStop = list(dict.fromkeys(externStop))



#Entfernen der xa0 in extern
for w in externStop:
    x = re.search("\xa0", w)
    if x:
        x = re.sub("\xa0",'',w)
        externStop.append(x)

with open("Quelldaten/Liste_Städte2.txt", "r",encoding='cp1252',errors='ignore') as input:
#with open("Schlagworte_Neu_28_03_2019_OhneFKZ2.txt", "r") as input:
    externStadt = input.read().split("\n") 



#Erweitern der Stoppwortliste
stop_words.extend(externStadt)
stop_words.extend(externStop) 




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
    #from nltk.tokenize import word_tokenize
    from nltk import word_tokenize
    nlp = spacy.load("de_core_news_lg")



    def applyregextoCopurs(KB, func, exceptionlist=[]):
        '''
        Function for seraching a corpus with a given matching function,
        which targets strings. It erases Entry which are found by the function
        KB - original Corpus
        func - (str) regexfunction to apply e.g re.findall("^([0-9]{2}:?)([A-Z])([A-Z])([0-9])", w)
        '''

        finds = []
        corpus = []
        for kb in KB:
            words = word_tokenize(kb)
            new_words = [w for w in words if not re.findall(func, w) or w in exceptionlist]
            finds.extend([w for w in words if re.findall(func, w) and w not in exceptionlist])
            corpus.append(' '.join(new_words))
        return finds, corpus

    # Findet traditionelle FKZ
    fkz, ListofSentences = applyregextoCopurs(ListofSentences, "^([0-9]{2}:?)([A-Z])([A-Z])([0-9])")

    # zahlen FKZ mit Buchstaben am Ende
    fkz2, ListofSentences = applyregextoCopurs(ListofSentences, '^[0-9]{7}[A-Z]?$')

    # Suche APs
    Aps_Stops, ListofSentences = applyregextoCopurs(ListofSentences, "^(AP)+.*[0-9]+$")




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
        
        #Kontroller dass hier alle Wörter getrennt wurden
        #for word in text:
            #print(word)
            
            
        if len(stopwords)>0:
            
            stop_words = list(map(lambda x: x.lower(), stopwords))
            
            text = [word for word in text if word not in stopwords]
            
        

        ##Stemming und Lemmatisation

        #lem = WordNetLemmatizer() # Vorigen Version des Processing
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



#-------------------------------------Rake Auswertung-----------------------------------
#Tutorial für Rake
#https://www.airpair.com/nlp/keyword-extraction-tutorial

#import RAKE as rake
#def Key_R_Old(string,stopwords):
#    rake_object = rake.Rake(stopwords)
#    keywords_R = rake_object.run(string)
#    return(keywords_R[:10])


#Dokumentaaion
#https://pypi.org/project/multi-rake/
def Key_R(string,stopwords1):
    r = Rake(min_chars=3,
        max_words=1,
        min_freq=1,
        language_code='de',  # 'en'
        stopwords=None,  # {'and', 'of'}
        lang_detect_threshold=50,
        max_words_unknown_lang=2,
        generated_stopwords_percentile=80,
        generated_stopwords_max_len=3,
        generated_stopwords_min_freq=2
        )
    keywords_R=r.apply(string)
    return(keywords_R[:10])


#------------------------------------textRankAuswertung------------------------------------

#Aus https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0
def Key_tr(string):
    #Muss am anfang initailiziert werden um nan fehler zu umgehen
    #tr4w = tcf.TextRank4Keyword()
    #tr4w.analyze(string, 
    #             candidate_pos = ['NOUN', 'PROPN'],
    #             window_size=4, 
    #             lower=False,
    #             stopwords=stop_words)
    #keywords_TR = tr4w.get_keywords(number=10)
    #return(keywords_TR)
    return 0





#---------------------------Automation Rake und TextRank---------------------------
#erzeugt aus eine PD Serie mit Texten einen Keywordsatz mit Rake und textrank
#In der PD Serie dürfen nur die Keywordsenthalten sein
#PD Seire sollte etwa so aussehen
#  0 blalbabla
#  1 blalbal
#  2 daoiajdoafk
#.....
#Es kann kein PD.Dataframe übergebnwerden
#stopwords solltena uch übergeben werden

def ErzeugeKeyW(PDSeries,stopwords):
    #Erzugen der Columns Namen für den Dataframe
    col = []
    col2 = []
    for y4 in range(1,13):
        #col.append('Rake_S'+str(y4))
        col2.append('TextR_S'+str(y4))
    col.extend(col2)
    
    #entfennen von Nan Values im CorpusDataframe
    PDSeries=PDSeries.dropna()
    
    
    #Erzeugen einer Liste, die die Lsiten aller Keywörter enthällt
    key_words_list =[]

    #Forloop über alle Texte
    for y in range(0,len(PDSeries)):
        token=[]
        
        #Aufruf Fukntionen um KEywörter zuz Erzuegen
        keywords_TR = Key_tr(PDSeries.iloc[y])
        #keywords_R = Key_R(PDSeries.iloc[y],stopwords)
        
        #Anhängen der Keywörter aus Textrank
        if len(keywords_TR)>=12:
            for z123 in range(0,12):
                    token.append(keywords_TR[z123][0])
        elif len(keywords_TR) < 12:
            for z123 in range(0,len(keywords_TR)):
                        token.append(keywords_TR[z123][0])
        #Anhänhen der Keywörter aus Rake
        #for x,y in keywords_R[:10]:
         #   token.append(x)
        #Zusammenfügen der Keywörter zu einem 20 Spalten langen Keywörtliste
        
        #Validieren ob es wirklich 20 Keywörter sind, sonst gibt es fehler im Dataframe
        #Anheften an die Keywordlist
        if len(token) ==12:    
            key_words_list.append(token)   
        else:
            for bla in range(0, 12-len(token)):
               token.append(None)
            key_words_list.append(token) 
    #print(key_words_list)
    df_return = pd.DataFrame(key_words_list,index=PDSeries.index ,columns=col)

    return(df_return)









def VergleichKeyW(corpus,Stich,n_index):
    '''
    # Suche Vorgegbene Keywörter im Coprus
    # Corpus bzw string wird mit Keywort list vergleichen. Gibt dabei alle Treffer
    # Aus der KEywortliste und das Keyword, mit welchen der Treffer erzeugt wurd aus.
    # Variable return_Keylist kann auch zurückgegeben werden, um alle Keywords, aus denen der Treffer
    # resultiert zurück zurgeben. 
    #

    # from nltk.tokenize import word_tokenize

    # Benötigt einen string ->corpus und eine Liste(Type Liste oder Array) mit Stichwörter zum Vergleich
    '''
    b1 = []
    Zwisch = []
    keywlist = []
    Gewicht = 0
    tok = word_tokenize(corpus)
    
    #Itteration über alle Keywörter in überegeben Stichwortverzeichnen
    for keyW in Stich:  
        for w in tok:
            #print(w)
            x = re.search(keyW, w)
            if (x):
                #print(x.string)
                if x not in Zwisch:
                    #Finden wie oft Keyword im Corpus vorkommt um gewicht herzustellen
                    Gewicht = len(re.findall(keyW, corpus))
                    zwisch1 = (x.string, Gewicht)
                    Zwisch.append(zwisch1)
                    keywlist.append(keyW)
                    #Zwisch.append(x.string)
    return_List = list(dict.fromkeys(Zwisch))
    
    '''Es kommt zu einem Keyerror wenn der vergleich keine Keywörter findet. Dann wird. Return_list leer und darüber kann
    nicht iteriert werden. Deswegen if bedingung. Um zu vermeiden das eine leere Liste übergeben wird
    '''   
    
    if return_List:
        #Zählen Wie häufig das Wort im Text vorkommt
        for t,v in  return_List:
            bla=corpus.count(t)
            b1.append((t, bla, v))   
        #Rückgabe KeyWortliste, als Liste
        dfObj = pd.DataFrame(b1)
        dfObj[3] = dfObj[2]*dfObj[1]
        dfObj = dfObj.sort_values(by=[3]).tail(10)
        dfObj = dfObj.set_index(0)
        returnListNeu = dfObj.index.tolist()
        
        #Rückgabe der AltStichworte
        keywlist = list(dict.fromkeys(keywlist))
        keywdf = pd.DataFrame(keywlist)
        keywdf[1]=1
        Keylist=keywdf.groupby(0).count().sort_values(by=[1]).tail(10)
        return_Keylist = Keylist.index.tolist()
        #Erzeugen der Columnanmes
        col = []
        col2 = []
        for y4 in range(1,11):
            col2.append('SE_S'+str(y4))
            col.append('AK_'+str(y4))
        
        
        #Füllen der Colummns wenn keine Einträge vorhanden sind mit Nonewerten
        if len(returnListNeu) < 10:
            for k in range(0,10-len(returnListNeu)):
                returnListNeu.extend(' ')
        if len(return_Keylist)<10:
            for k in range(0,10-len(return_Keylist)):
                return_Keylist.extend(' ')
        
        #Erzeugen der Rückgabe DF bestehend aus den Stichworten aus den 3 Funktionen
        df = pd.DataFrame(np.array(returnListNeu).reshape(1,10), columns = col2, index = n_index)
        df2 = pd.DataFrame(np.array(return_Keylist).reshape(1,10), columns = col, index = n_index)
        result = pd.concat([df, df2], axis=1)
    else:
    #Übergabe eines Leeren Dataframes ohne Indexfehler
        col = []
        col2 = []
        for y4 in range(1,11):
            col2.append('SE_S'+str(y4))
            col.append('AK_'+str(y4))
        
        df = pd.DataFrame(columns = col2, index= n_index)
        df2 = pd.DataFrame(columns = col, index= n_index)
        result = pd.concat([df, df2], axis=1)
    
    return(result)


def dfToExcel(dataframe, string):
    name=string+'.xlsx'
    writer = pd.ExcelWriter(name)
    dataframe.to_excel(writer,'Sheet1')
    writer.save()





def LoadKurzBeschreibungen():

    print('Starting...')

    # Öffnen der Quelldatei ,ot Lurzbeschreibungen
    xlsx_file = pd.ExcelFile(
        'Quelldaten\FKZ_Kurzbeschreibung_alle_Phasen.xlsx')  # Wird in Excel Erstellt. Ist nur eine Liste der FKZ die mit der Nummerierung mit dem Tokenizer übereinstimmt!
    KurzB = xlsx_file.parse('Kurzbeschreibungen')

    FKZ_proj_Phasen = {}

    Phasen = ['Phase1', 'Phase2', 'Phase3']
    for Phase in Phasen:
        FKZ_proj_Phasen[Phase] = xlsx_file.parse(Phase)

    # Einfügen der abgeschlossnen FKZ!!!
    # Wird in Excel Erstellt. Ist nur eine Liste der FKZ die mit der Nummerierung mit dem Tokenizer übereinstimmt!

    FKZ_Tokenizer_FB_abgeschlossen = {}

    FKZ_Tokenizer_FB_abgeschlossen['Phase2'] = pd.ExcelFile('Quelldaten\Phase2_schlagwortFKZ.xlsx').parse(
        'BF_Fragebogen_2.10.2019_11.50')
    FKZ_Tokenizer_FB_abgeschlossen['Phase3'] = pd.ExcelFile('Quelldaten\Phase3_schlagwortFKZ.xlsx').parse(
        'BF_Fragebogen_2.10.2019_11.52')

    # Einfügen von Phase 2 und 3 alle Stichworte die nicht FB abgeschlossen haben
    # Wird in Excel Erstellt. Ist nur eine Liste der FKZ die mit der Nummerierung mit dem Tokenizer übereinstimmt!

    FKZ_Tokenizer_FB_Nichtabgeschlossen = {}

    FKZ_Tokenizer_FB_Nichtabgeschlossen['Phase2'] = pd.ExcelFile(
        'Quelldaten\Phase2_nichtabgeschlossen_alleStich.xlsx').parse('Phase2_nichtabgeschlossen_alleS')
    FKZ_Tokenizer_FB_Nichtabgeschlossen['Phase3'] = pd.ExcelFile(
        'Quelldaten\Phase3_nichtabgeschlossen_alleStich.xlsx').parse('Phase3_nichtabgeschlossen_alleS')

    # Vergleicht zwei Colums und gibt Dataframe mit übereinstimmung zurück

    Phasen = ['Phase1', 'Phase2', 'Phase3']

    KurzB_ausPhase = {}

    for Phase in Phasen:
        KurzB_ausPhase[Phase] = KurzB[KurzB['Förderkennzeichen'].isin(FKZ_proj_Phasen[Phase]['Fkz'])]

    # Erstellen MetaDF für Validierung
    valid_meta = [{'01_Phase': 'Phase1',
                   '02_Einträge Alle aus Phase': len(FKZ_proj_Phasen['Phase1']),
                   '03_Nicht Abgeschlossen FB': None,
                   '04_Abgeschlossen FB': None,
                   '05_Beschreibungen Vorhanden aus Phase': len(KurzB_ausPhase['Phase1'])},
                  {'01_Phase': 'Phase2',
                   '02_Einträge Alle aus Phase': len(FKZ_proj_Phasen['Phase2']),
                   '03_Nicht Abgeschlossen FB': len(FKZ_Tokenizer_FB_Nichtabgeschlossen['Phase2']),
                   '04_Abgeschlossen FB': len(FKZ_Tokenizer_FB_abgeschlossen['Phase2']),
                   '05_Beschreibungen Vorhanden aus Phase': len(KurzB_ausPhase['Phase2'])},
                  {'01_Phase': 'Phase3',
                   '02_Einträge Alle aus Phase': len(FKZ_proj_Phasen['Phase3']),
                   '03_Nicht Abgeschlossen FB': len(FKZ_Tokenizer_FB_Nichtabgeschlossen['Phase3']),
                   '04_Abgeschlossen FB': len(FKZ_Tokenizer_FB_abgeschlossen['Phase3']),
                   '05_Beschreibungen Vorhanden aus Phase': len(KurzB_ausPhase['Phase3'])}]

    df_valid_meta = pd.DataFrame.from_dict(valid_meta)

    # Is not in!! HErausfinden. Filtern der nicht vorhandenen FKZ
    # Projekte mit Schlagwörter werden gefiltert

    KurzB_ausPhase['Phase2'] = KurzB_ausPhase['Phase2'][
        ~KurzB_ausPhase['Phase2']['Förderkennzeichen'].isin(FKZ_Tokenizer_FB_Nichtabgeschlossen['Phase2']['FKZ'])]
    KurzB_ausPhase['Phase3'] = KurzB_ausPhase['Phase3'][
        ~KurzB_ausPhase['Phase3']['Förderkennzeichen'].isin(FKZ_Tokenizer_FB_Nichtabgeschlossen['Phase3']['FKZ'])]


    print('Bereite Daten vor...')
    # Entfernen von leeren Beschreibungen
    Phasen = ['Phase1', 'Phase2', 'Phase3']
    for Phase in Phasen:
        # Entferne Leerzeilen, Duplikate, Resete Index
        KurzB_ausPhase[Phase] = KurzB_ausPhase[Phase].dropna(subset=['Kurzbeschreibung'])
        KurzB_ausPhase[Phase] = KurzB_ausPhase[Phase].drop_duplicates(subset=['Kurzbeschreibung'])
        KurzB_ausPhase[Phase] = KurzB_ausPhase[Phase].reset_index(drop=True)
        KurzB_ausPhase[Phase] = KurzB_ausPhase[Phase].set_index('Förderkennzeichen')


    print('Loaded Kurzbeschreibungen...')

    return  KurzB_ausPhase





def Keyworderzeugung(SavetoExcel = False, Preprocess = False, Fuzzy=False, FuzzyRank=False,
                     _reversed = False, Remove_specCar = False, IgnoreWord_list =[None]):

    print('Starting...')

    #Öffnen der Quelldatei ,ot Lurzbeschreibungen
    xlsx_file = pd.ExcelFile('Quelldaten\FKZ_Kurzbeschreibung_alle_Phasen.xlsx')#Wird in Excel Erstellt. Ist nur eine Liste der FKZ die mit der Nummerierung mit dem Tokenizer übereinstimmt!
    KurzB = xlsx_file.parse('Kurzbeschreibungen')

    FKZ_proj_Phasen = {}

    Phasen =['Phase1','Phase2','Phase3']
    for Phase in Phasen:
        FKZ_proj_Phasen[Phase] = xlsx_file.parse(Phase)


    #Einfügen der abgeschlossnen FKZ!!!
    #Wird in Excel Erstellt. Ist nur eine Liste der FKZ die mit der Nummerierung mit dem Tokenizer übereinstimmt!

    FKZ_Tokenizer_FB_abgeschlossen = {}

    FKZ_Tokenizer_FB_abgeschlossen['Phase2'] = pd.ExcelFile('Quelldaten\Phase2_schlagwortFKZ.xlsx').parse('BF_Fragebogen_2.10.2019_11.50')
    FKZ_Tokenizer_FB_abgeschlossen['Phase3'] = pd.ExcelFile('Quelldaten\Phase3_schlagwortFKZ.xlsx').parse('BF_Fragebogen_2.10.2019_11.52')

    #Einfügen von Phase 2 und 3 alle Stichworte die nicht FB abgeschlossen haben
    #Wird in Excel Erstellt. Ist nur eine Liste der FKZ die mit der Nummerierung mit dem Tokenizer übereinstimmt!

    FKZ_Tokenizer_FB_Nichtabgeschlossen = {}

    FKZ_Tokenizer_FB_Nichtabgeschlossen['Phase2'] = pd.ExcelFile('Quelldaten\Phase2_nichtabgeschlossen_alleStich.xlsx').parse('Phase2_nichtabgeschlossen_alleS')
    FKZ_Tokenizer_FB_Nichtabgeschlossen['Phase3'] = pd.ExcelFile('Quelldaten\Phase3_nichtabgeschlossen_alleStich.xlsx').parse('Phase3_nichtabgeschlossen_alleS')



    #Vergleicht zwei Colums und gibt Dataframe mit übereinstimmung zurück

    Phasen = ['Phase1', 'Phase2', 'Phase3']

    KurzB_ausPhase = {}

    for Phase in Phasen:
        KurzB_ausPhase[Phase] = KurzB[KurzB['Förderkennzeichen'].isin(FKZ_proj_Phasen[Phase]['Fkz'])]


    #Erstellen MetaDF für Validierung
    valid_meta=[{'01_Phase':'Phase1',
                 '02_Einträge Alle aus Phase': len(FKZ_proj_Phasen['Phase1']),
                 '03_Nicht Abgeschlossen FB': None,
                 '04_Abgeschlossen FB': None,
                 '05_Beschreibungen Vorhanden aus Phase': len(KurzB_ausPhase['Phase1'])},
                {'01_Phase':'Phase2',
                 '02_Einträge Alle aus Phase': len(FKZ_proj_Phasen['Phase2']),
                 '03_Nicht Abgeschlossen FB': len(FKZ_Tokenizer_FB_Nichtabgeschlossen['Phase2']),
                 '04_Abgeschlossen FB': len(FKZ_Tokenizer_FB_abgeschlossen['Phase2']),
                 '05_Beschreibungen Vorhanden aus Phase': len(KurzB_ausPhase['Phase2'])},
                 {'01_Phase':'Phase3',
                 '02_Einträge Alle aus Phase': len(FKZ_proj_Phasen['Phase3']),
                 '03_Nicht Abgeschlossen FB': len(FKZ_Tokenizer_FB_Nichtabgeschlossen['Phase3']),
                 '04_Abgeschlossen FB': len(FKZ_Tokenizer_FB_abgeschlossen['Phase3']),
                 '05_Beschreibungen Vorhanden aus Phase': len(KurzB_ausPhase['Phase3'])}]

    df_valid_meta = pd.DataFrame.from_dict(valid_meta)


    # Is not in!! HErausfinden. Filtern der nicht vorhandenen FKZ
    # Projekte mit Schlagwörter werden gefiltert

    KurzB_ausPhase['Phase2'] = KurzB_ausPhase['Phase2'][~KurzB_ausPhase['Phase2']['Förderkennzeichen'].isin(FKZ_Tokenizer_FB_Nichtabgeschlossen['Phase2']['FKZ'])]
    KurzB_ausPhase['Phase3'] = KurzB_ausPhase['Phase3'][~KurzB_ausPhase['Phase3']['Förderkennzeichen'].isin(FKZ_Tokenizer_FB_Nichtabgeschlossen['Phase3']['FKZ'])]

    df_valid_meta['06_Projekte ohne Schlagwörter je Phase']=[len(KurzB_ausPhase['Phase1']),len(KurzB_ausPhase['Phase2']),len(KurzB_ausPhase['Phase3'])]

    print('Bereite Daten vor...')
    #Entfernen von leeren Beschreibungen
    Phasen = ['Phase1', 'Phase2', 'Phase3']
    for Phase in Phasen:
        #Entferne Leerzeilen, Duplikate, Resete Index
        KurzB_ausPhase[Phase] = KurzB_ausPhase[Phase].dropna(subset=['Kurzbeschreibung'])
        KurzB_ausPhase[Phase] = KurzB_ausPhase[Phase].drop_duplicates(subset=['Kurzbeschreibung'])
        KurzB_ausPhase[Phase] = KurzB_ausPhase[Phase].reset_index(drop=True)
        KurzB_ausPhase[Phase] = KurzB_ausPhase[Phase].set_index('Förderkennzeichen')


        if Preprocess:
            ind = KurzB_ausPhase[Phase].index
            col = KurzB_ausPhase[Phase].columns

            data = PrePross(KurzB_ausPhase[Phase]['Kurzbeschreibung'].values.tolist(),Fuzzy, FuzzyRank, _reversed, Remove_specCar,IgnoreWord_list)
            KurzB_ausPhase[Phase] = pd.DataFrame(data, index=ind, columns=['Kurzbeschreibung'])
            KurzB_ausPhase[Phase].index.name = 'Förderkennzeichen'




    df_valid_meta['07_Projekte ohne Schlagwörternach Leer und Doppelen']=[len(KurzB_ausPhase['Phase1']),len(KurzB_ausPhase['Phase2']),len(KurzB_ausPhase['Phase3'])]

    if SavetoExcel:
        df_valid_meta.to_csv('Meta.csv',sep=';')





    #Einfügen der alten Stichworte
    df_altStich = pd.read_csv('Quelldaten\MiestverwendeteStichworteNetz.csv')
    neu = df_altStich[df_altStich['degree'] >= 20]
    AltStich = neu['Label'].values

    AltStich = np.append(AltStich, ['erneuer'], 0)



    def KeyvonPhase(corpus,stopworte,keywords):
        '''
        Funktion für Keyword erzuegen. steht in der Ausfführung da speziel für df_kb_phase
        angefertigt. kann aber für jede beliebige df die genauso aufgebaut ist ausgeführt werden
        benötigt eine Pands.series als Corpus wo zu analysierende Strings enthalten sind
        z.b.
        0 bla
        1 blabla
        2 blablala
                          sw sind Stopwörds!!!!
        '''


        KeyWdf = ErzeugeKeyW(corpus,stopworte) #Erzeuge Keywords


        corpus = corpus.dropna() #Entfernen von Nan


        #Erzeugen eines DAtaframes für Ausgabe von Vergleich Keywords
        col = []
        col2 = []
        #print('Aufruf')
        for y4 in range(1,11):
            col2.append('SE_S'+str(y4))
            col.append('AK_'+str(y4))
        #corpus[1].index

        #Erzuegen des Dataframes mit Stichwortvergleich, siehe Aufruf VergleichKeyW
        df_KeyWVergl = pd.DataFrame(columns=col)
        for j in range(0,len(corpus)):
            df_KeyWVergl = pd.concat([df_KeyWVergl,VergleichKeyW(corpus.iloc[j],keywords, n_index =[corpus.index[j]])])


        #final_result = pd.DataFrame()
        final_result = pd.concat([KeyWdf,df_KeyWVergl],axis=1)

        #Transponieren und Entfernung von Duplikaten
        fr_clear = final_result.transpose()

        for i in fr_clear.columns:
            c = fr_clear[i].duplicated()
            fr_clear[i].loc[c] = None

        fr_clear = fr_clear.transpose()

        return(fr_clear)



    #Keyword Erzeugung, siehe Funktion hierüber

    KeywordDic = {}

    print('ErzeugeKeywords...')

    for Phase in Phasen:
        KeywordDic[Phase] = KeyvonPhase(KurzB_ausPhase[Phase]['Kurzbeschreibung'], stop_words, AltStich)

    print('Keywordsuche zuende...')

    '''Falls nur externStopTextrank ErzeugeKeyW(KurzB_ausPhase['Phase1']['Kurzbeschreibung'], stop_words) !'''

    if SavetoExcel:
    #Abspeichern
        dfToExcel(KeywordDic['Phase1'],'Ergebnis\Keywords_Phase1')
        dfToExcel(KeywordDic['Phase2'],'Ergebnis\Keywords_Phase2_NichtAusgefüllt')
        dfToExcel(KeywordDic['Phase3'],'Ergebnis\Keywords_Phase3_Nichtausgefüllt')
        dfToExcel(df_valid_meta,'Ergebnis\MetaValidierungsMatrix')


        df_P2_diff_abg_nabg = FKZ_Tokenizer_FB_Nichtabgeschlossen['Phase2'][~FKZ_Tokenizer_FB_Nichtabgeschlossen['Phase2']['FKZ'].isin(FKZ_Tokenizer_FB_abgeschlossen['FKZ'])]
        df_P3_diff_abg_nabg = FKZ_Tokenizer_FB_Nichtabgeschlossen['Phase3'][~FKZ_Tokenizer_FB_Nichtabgeschlossen['Phase3']['FKZ'].isin(FKZ_Tokenizer_FB_abgeschlossen['FKZ'])]

        dfToExcel(df_P2_diff_abg_nabg,'Ergebnis\Keywords_Phase2_nichtAbgeschlossen')
        dfToExcel(df_P3_diff_abg_nabg,'Ergebnis\Keywords_Phase3_nichtAbgeschlossen')

    return KeywordDic, df_valid_meta, FKZ_Tokenizer_FB_abgeschlossen



if __name__ == "__main__":
    KeywordDic_Enargus, df_valid_meta, KeywordDic_FB = Keyworderzeugung()




#------------------------------------Word2Vec----------------------------------------
#https://github.com/bguvenc/keyword_extraction/blob/master/get_keyword.py
#Funktioniert wegen Package nicht

#-----------------------------------------------------------------------------------

#Word Embedding
#https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795
#https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/
#http://ai.intelligentonlinetools.com/ml/k-means-clustering-example-word2vec/
#https://towardsdatascience.com/google-news-and-leo-tolstoy-visualizing-word2vec-word-embeddings-with-t-sne-11558d8bd4d


#CountVectorizeer
#http://kavita-ganesan.com/extracting-keywords-from-text-tfidf/#.XNA-t8Tgrcs

