import pandas as pd
import pickle
from KeywordAlgorithms import tfIdf_Vectorizer_getKeyWords, tfIdf_Vectorizer_train, getTextrankKeywords
import Network_Preprocessing as n_p
import spacy

'''
saveFile = True
CreatePairs = True          # Create Pairs for network
usePrePross = False
loadPairs = False           # Load Pairs from Excel
PrePross = True             # Use Preprsoceesing
bigrams = True              #Create bigram data with TFIDF
useBigramData = False        #Use data for futher Process
ZahlStichw = 20             # Number of Keywords to generate from TfIDF

# Which Algor to use
TEXTRANK = False
TFIDF = False
TFIDFModi = True            # Best and Actual Method
TFIDFModi2 = False #Prototype, doesnt work properly
'''

saveFile = False
CreatePairs = True          # Create Pairs for network
usePrePross = False
loadPairs = False           # Load Pairs from Excel
PrePross = False             # Use Preprsoceesing
bigrams = True              #Create bigram data with TFIDF
useBigramData = False        #Use data for futher Process
ZahlStichw = 20             # Number of Keywords to generate from TfIDF

# Which Algor to use
TEXTRANK = False
TFIDF = False
TFIDFModi = True            # Best and Actual Method
TFIDFModi2 = False #Prototype, doesnt work properly


Plot = False                # Create tnse with networkx
network = False             #Create networkx graph



# Package for German Preprocessing https://github.com/jfilter/german-preprocessing


type_=''

from nltk.corpus import stopwords

stop_words = stopwords.words("german")  # orginal ist in "english", kann aber auch mit Deutsch
stop_words.append(' ')
stop_words.append('``')
stop_words.append('""')
# Hier weitere stopwörter anhängen falls gebraucht
stop_words.extend(('sollen', 'mit', 'sowie', 'anhand',
                   'darüber', 'hinaus', 'ersten', 'zweiten',
                   'dritten', 'dabei', 'dass', 'darauf', 'bzw', 'deren',
                   'derer', 'solche', 'solch', 'eigenen', 'einen', 'eine',
                   'einen'))
# Spezielle Stopwöret für die Keywordextraktion
stop_words.extend(('ziel',
                   'vorhaben', 'vorhabens',
                   'schritt',
                   'basis',
                   'projekt', 'projekts',
                   'fokus',
                   'liegt',
                   'Arbeitspaket',
                   'gestellt',
                   'hinsichtlich',
                   'neuen',
                   'mittels',
                   'z.B.', 'Arbeitspacket', 'AP',
                   'TUM',
                   'FH',
                   'TH',
                   'Kosten',
                   'Gesamtkosten',
                   'Forschung',
                   'Vorhaben',
                   'Plan',
                   'ISE', 'GENESIS', 'PERC', '%', 'TUK', 'TU', 'ISFH', 'PERC+', 'a2-solar', 'B.', 'DI', 'HPDI',
                   'TU-München', 'Tegtmeyer','mw','kw','ag','zb','Kcal'
                   ))


#Zusatzstopwords
stop_words.extend(['m²','FKZ','VE','/','„','KG','Co.','RAG','DMT','Ptj','K','PHI','u.a.','u.ä.','DUE','B.&S.U.',
                   'ITC','ISE','ERC','m','Ost','Süd','EASD','A.3','B.3','LES','TILSE','IAP','LWS','HEM','8)',
                   'Test','FTA','FH','A1','ITI','Nord','West','Ruhr','OPVT','m2','HH','HTW','TT-WB/ENS1','TT/MKX',
                   'ILK','m³','Standort','Adlershof','EBC','EEC','OPV','IKT','GSG','Ort','PCS','Projektabschnitt',
                   'kWh/m²','Ergebnisse','Voraussetzung','ScenoCalc','Schritt','Anbieter','Glaubwürdigkeit','Bezug',
                   'Validierung','Gebiet','Verbundprojekt','Equipment','Fragestellungen','Einfluss','Auswahl',
                   'Relation','Indevo','Projektpartnern','Anzahl','Angebot','Bedarf', 'Ch','Dec','Dfki',
                   'Ee','Ens','Et','Fa','lt','Ksb','Kw','Nvz','Of','Shc','Sol','Swt','Ts','Tum','Tud','Tm', 'Ttwb','Tvb',
                   'Uvb','Ap','Arbeitspaket', 'Frauenhofer','Firma','Konzept','Covestro','Fresnel','rwth'])


# Load extern Stopword Data

f = open("Quelldaten//StopWords_Algo.txt", "r", encoding='cp1252')
extdata = f.read()
extdata = extdata.replace('\n','')
extdata = extdata.split(',')
stop_words.extend(extdata)

with open("Quelldaten//Liste_Städte2.txt", "r",encoding='cp1252',errors='ignore') as input:
    externStadt = input.read().split("\n") 
    
stop_words.extend(externStadt)




if __name__ == "__main__":

    if CreatePairs:
        
        
        print('Creating Text')
        xlsx_file = pd.ExcelFile('Quelldaten//EnargusDaten_Vollständig_2_29_04_2019.xlsx')
        dfEn = xlsx_file.parse('Sheet1')
        dfEn.set_index('Förderkennzeichen', inplace=True)


        dfEn['Beschreibung'] = dfEn['Beschreibung'].fillna(' ')
        dfEn_BackUp = dfEn.copy()
        dfEn = dfEn[dfEn['Beschreibung'] != ' ']

        FKZ = dfEn.index.values.tolist()
        
        filename = 'FKZ.sav'
        pickle.dump(FKZ, open(filename, 'wb'))
        

        Arr_KeyWord_df_pre = dfEn['Beschreibung'].values.tolist()[0:50]
        
        
        print('Länge Corpus', len(Arr_KeyWord_df_pre))

        word_list = ['bla']

        if PrePross:
            Arr_KeyWord_df_New = n_p.PrePross(Arr_KeyWord_df_pre, _comma=False,
                                              Fuzzy=False,
                                              FuzzyRank=True,
                                              _reversed=True,
                                              Remove_specCar=True, 
                                              IgnoreWord_list=word_list, 
                                              stem=True,
                                              stopwords=stop_words)


            if saveFile:
                filename = 'Working_Corpus.pkl'
                pickle.dump(Arr_KeyWord_df_New, open(filename, 'wb'))

            '''
            Arr_KeyWord_df_New =
            [text1,
            text2,
            text3]

            text = string ohne Kommas aber mit Punkttrennung 
            '''
        else:
            Arr_KeyWord_df_New = Arr_KeyWord_df_pre
        
        
        
        if usePrePross:
            Arr_KeyWord_df_New = pickle.load(open('KeywordsOutput/PrePross_Corpus.pkl','rb'))
            
            
        print('Creating Keywords..')






        if TEXTRANK:

            type_ = 'TextRank'
            kw = []

            # def f(x):
            #   return x.replace(',', ' ')

            # dfEn['Beschreibung'] = dfEn['Beschreibung'].apply(lambda x: f(x))
            # Arr_KeyWord_df = dfEn['Beschreibung'].values.tolist()[1:100]
            # Arr_KeyWord_df_pre = Arr_KeyWord_df
            corpus = Arr_KeyWord_df_New

            kw = [getTextrankKeywords(entry, stop_words) for entry in corpus]
            print('Created TR Keywords...')


            def genKw(corpus):
                for x in range(len(corpus)):
                    projekt = []
                    for i in range(len(corpus[x])):
                        word = corpus[x][i][0]
                        projekt.append(word)
                    yield ','.join(map(str, projekt))


            kw_v2 = list(genKw(kw))

            if saveFile:
                filename = 'KeywordArray_TextRank.pkl'
                pickle.dump(kw_v2, open(filename, 'wb'))

        if TFIDF:

            type_ = 'TFIDF'

            # def f(x):
            #    return x.replace(' ', ',')

            # dfEn['Beschreibung'] = dfEn['Beschreibung'].apply(lambda x: f(x))

            # Arr_KeyWord_df = dfEn['Beschreibung'].values.tolist()
            # Arr_KeyWord_df_pre = Arr_KeyWord_df_New

            corpus = Arr_KeyWord_df_New

            if PrePross:
                #wörter werden in Prepros capitalized
                stop_words = list(map(lambda x: x.capitalize(), stop_words))


            tfdf = tfIdf_Vectorizer_train(corpus,
                                          ngrams = False,
                                          stop_word = stop_words,
                                          ngram_range=(1,2))
            kw = []

            kw = [tfIdf_Vectorizer_getKeyWords(tfdf, i, n=ZahlStichw).index.tolist() for i in range(0, len(corpus))]


            def genKw(corpus):
                for entry in corpus:
                    bagofwords = ','.join(map(str, entry))
                    yield bagofwords


            kw_v2 = list(genKw(kw))

            if saveFile:
                filename = 'KeywordArray_TFIDF.sav'
                pickle.dump(kw_v2, open(filename, 'wb'))




        if TFIDFModi:
            
            type_ = 'TFIDFModi'
            corpus = []
            nlp = spacy.load("de_core_news_lg")


            def corpusgen(func_corpus):
                for entry in func_corpus:
                    doc = nlp(entry)
                    newEntry = []
                    for w in doc:
                        if w.pos_ == 'NOUN':
                            newEntry.append(w.text) #Lemmatization
                    yield newEntry


            corpus = list(corpusgen(Arr_KeyWord_df_New))

            corpus = [' '.join(x) for x in corpus]

            tfdf = tfIdf_Vectorizer_train(corpus, 
                                          standard=True,
                                          ngrams=False, 
                                          stop_word = stop_words, 
                                          ngram_range=(1,2))

            
            kw = []

            kw = [tfIdf_Vectorizer_getKeyWords(tfdf, i, n=ZahlStichw).index.tolist() for i in range(0, len(corpus))]
            
            kw_Val = [tfIdf_Vectorizer_getKeyWords(tfdf, i, n=ZahlStichw).values for i in range(0, len(corpus))]

            def genKw(corpus):
                for entry in corpus:
                    bagofwords = ','.join(map(str, entry))
                    yield bagofwords


            kw_v2 = list(genKw(kw))

            if saveFile:
                filename = 'KeywordArray_TFIDFModi.sav'
                pickle.dump(kw_v2, open(filename, 'wb'))
                filename = 'TFIDFModi_MatrixValues.sav'
                pickle.dump(kw_Val, open(filename, 'wb'))
                
                
            if bigrams:                

                tfdf = tfIdf_Vectorizer_train(corpus, 
                              standard=True,
                              ngrams=True, 
                              stop_word = stop_words, 
                              ngram_range=(2,3))
    
                
                bikw = []
    
                bikw = [tfIdf_Vectorizer_getKeyWords(tfdf, i, n=ZahlStichw).index.tolist() for i in range(0, len(corpus))]
                
                bikw_Val = [tfIdf_Vectorizer_getKeyWords(tfdf, i, n=ZahlStichw).values for i in range(0, len(corpus))]
    
                def genKw(corpus):
                    for entry in corpus:
                        bagofwords = ','.join(map(str, entry))
                        yield bagofwords
    
    
                bikw_v2 = list(genKw(kw))
                
                from sklearn.feature_extraction.text import CountVectorizer
                
                cv = CountVectorizer(max_df=0.85, 
                                     ngram_range=(1,1),
                                     stop_words=stop_words)
                word_count_vector = cv.fit_transform(corpus)
            
                # you only needs to do this once, this is a mapping of index to
                feature_names = cv.get_feature_names()
                
                cv = CountVectorizer(max_df=0.85, 
                                     ngram_range=(2,3),
                                     stop_words=stop_words)
                word_count_vector_bi = cv.fit_transform(corpus)
            
                # you only needs to do this once, this is a mapping of index to
                feature_names_bi = cv.get_feature_names()
                
                if useBigramData:
                    kw_v2 = bikw_v2
               
    
                if saveFile:
                    filename = 'BigramKWArray_TFIDFModi.sav'
                    pickle.dump(bikw_v2, open(filename, 'wb'))
                    filename = 'Bigram_TFIDFModi_MatrixValues.sav'
                    pickle.dump(bikw_Val, open(filename, 'wb'))
                    filename = 'Bigram_PandasData.sav'
                    pickle.dump(tfdf, open(filename, 'wb'))
                    filename = 'Countvektor.sav'
                    pickle.dump(word_count_vector, open(filename, 'wb'))
                    filename = 'Countvektor_Bigram.sav'
                    pickle.dump(word_count_vector_bi, open(filename, 'wb'))
                    filename = 'feature_names.sav'
                    pickle.dump(feature_names, open(filename, 'wb'))
                    filename = 'feature_names_Bigram.sav'
                    pickle.dump(feature_names_bi, open(filename, 'wb'))
                    
                
        if TFIDFModi2:
            
            type_ = 'TFIDFModi2'
            corpus = []
            nlp = spacy.load("de_core_news_lg")


            def corpusgen(func_corpus):
                for entry in func_corpus:
                    doc = nlp(entry)
                    newEntry = []
                    for w in doc:
                        if w.pos_ == 'NOUN':
                            newEntry.append(w.text)
                    yield newEntry


            corpus = list(corpusgen(Arr_KeyWord_df_New))

            corpus = [' '.join(x) for x in corpus]


            names,tfidf_vec = tfIdf_Vectorizer_train(corpus, 
                                          standard=False,
                                          ngrams=True, 
                                          stop_word = stop_words, 
                                          ngram_range=(1,2))


            def KWGen(names,tfidf_vec, n=ZahlStichw):
                #Keywordgenerator, der direkt mit einer Sparse Matrix funktioniert
                for x in range(tfidf_vec.shape[0]):
                    
                    df = pd.DataFrame(tfidf_vec[x,:].todense().tolist()[0], index=names)
                    yield df.sort_values(0,ascending=False)[:n].index.tolist()
                  
            
            
            kw = list(KWGen(names,tfidf_vec,n=ZahlStichw))


            def genKw(corpus):
                for entry in corpus:
                    bagofwords = ','.join(map(str, entry))
                    yield bagofwords


            kw_v2 = list(genKw(kw))
           

            #import gensim
            #from gensim import corpora
            #from gensim.utils import simple_preprocess
            
           # mydict = corpora.Dictionary([simple_preprocess(entry) for entry in corpus])
            #corpus = [mydict.doc2bow(simple_preprocess(entry)) for entry in corpus]
            
            #tfidf = models.TfidfModel(corpus, smartirs='ntc')
            
            #for doc in tfidf[corpus]:
            #    print([[mydict[id], np.around(freq, decimals=2)] for id, freq in doc])
    

        
           

            if saveFile:
                filename = 'KeywordArray_TFIDFModi2.sav'
                pickle.dump(kw_v2, open(filename, 'wb'))
                
            

        from NLP import Countmatrix, GetPairsNumpy #, GetPairsWithWeight
        print('Creating CountMatrix')
        
        KeyM = Countmatrix(kw_v2, pandas=False, matrix=True)
        
        filename = 'SparseMatrix_TFIDFModi.sav'
        pickle.dump(KeyM, open(filename, 'wb'))

        Edges = GetPairsNumpy(KeyM)

        # KeyM = pd.DataFrame(KeyM[0].todense(), columns=KeyM[1])
        Edges = pd.DataFrame(Edges, columns=['Source', 'Target', 'Projekt']).rename_axis('Index')

        if saveFile:
            if TEXTRANK or TFIDF or TFIDFModi or TFIDFModi2:
                Edges.to_csv('KeywordsOutput//Complete_edges_' + str(type_) + '.csv', sep=';', encoding='cp1252')
            else:
                Edges.to_csv('KeywordsOutput//Complete_edges.csv', sep=';', encoding='cp1252')

    if loadPairs:
       
        if TEXTRANK:
            type_ = 'TextRank'
        elif TFIDF:
            type_ = 'TFIDF'
        elif TFIDFModi:
            type_ = 'TFIDFModi'
        if type_:
            Edges = pd.read_csv('KeywordsOutput//Complete_edges_' + str(type_) + '.csv', sep=';', encoding='cp1252')
            Edges.drop('Index', axis=1, inplace=True)
            print('Edges geladen')

        else:

            Edges = pd.read_csv('KeywordsOutput//Complete_edges.csv', sep=';', encoding='cp1252')
            Edges.drop('Index', axis=1, inplace=True)


    if network:
        '''Fals ein Network ausgegeben werden soll, network auf True stellen'''
        # Siehe https://www.kaggle.com/ferdzso/knowledge-graph-analysis-with-node2vec
        # siehe https://orbifold.net/default/node2vec-embedding/
        import networkx as nx

        print('Creating Graph and TSNE')
        
        G = nx.Graph()

        source = Edges['Source'].values
        target = Edges['Target'].values

        M = [source, target]


        def creNode(M):
            for i in range(0, len(M[0])):
                yield M[0][i], M[1][i]


        gen = creNode(M)
        for pair in gen:
            G.add_edge(pair[0], pair[1])

        from node2vec import Node2Vec

        n2v_obj = Node2Vec(G, dimensions=10, walk_length=5, num_walks=10, p=1, q=1, workers=1)
        n2v_model = n2v_obj.fit(window=3, min_count=1, batch_words=4)
        node_embeddings = n2v_model.wv.vectors
        node_ids = n2v_model.wv.index2word

        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2)

        node_embeddings_2d = tsne.fit_transform(node_embeddings)

        filename = 'TSNE_Embeddings_'+str(type_)+'.sav'
        pickle.dump(node_embeddings_2d, open(filename, 'wb'))

        if Plot:
            import matplotlib.pyplot as plt

            plt.subplot(111)

            # pos = nx.spring_layout(G, k=1.15, iterations=100)
            pos = nx.layout.spring_layout(H)
            nx.draw(G, with_labels=False, node_size=0.1, width=0.05)
            plt.savefig('Network.png', dpi=400)
            plt.show()

