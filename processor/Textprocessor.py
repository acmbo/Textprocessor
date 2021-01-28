import pandas as pd
import pickle
from processor.extractionAlgorithms import tfIdf_Vectorizer_getKeyWords,tfIdf_Vectorizer_train,getTextrankKeywords
from processor import preprocessing as pre


class Textprocessor:

    def __init__(self, corpus, stopwords=None,wordexclusionlist=None):
        '''
        Class for Textprocessing.
        Generates Keywords, can create a Network from the Keywords

        :param corpus: list - Textcorpus, which Keywords will be extract from. Should be a list of Texts, e.g ['one or two','three or four'] (Not a list of Words)
        :param stopwords: list - Words, which should be excluded by the Preprocessing
		:param wordexclusionlist: list - Words, which will be excluded from Preprocessing when searching for Words (Wörter werden im Corpus gehalten und nicht gefiltert)
		'''
        self.Corpus = corpus
        self.stop_words = stopwords
        self.word_exclusion_list = wordexclusionlist
        self.PreProssTxT = None     # Werden in Funktionen assigned
        self.Keywords = None        # Werden in Funktionen assigned
        self.Edges = None           # Werden in Funktionen assigned
        self.Graph = None           # Werden in Funktionen assigned

        print('======================================')
        print('Length of Corpus: ',len(self.Corpus))

        if stopwords:
            print('Stopwords given')

        else:
            print('No Stopwords given. Using premade Stopwords')
            from nltk.corpus import stopwords
            stopwords = stopwords.words("german")  # orginal ist in "english", kann aber auch mit Deutsch
            self.stop_words = stopwords

        print('======================================')


    def preprossText(self, Remove_specCar=True):
        '''
        Preprocessing of the given Corpus in the Class
        Function removes punctuation, but leaves Endpoints '.', because the algorithms using these to mark
        Sentence Ending.

        Included Preprocessing:
            - Part of Speech Tagging
            - Lemmatazation
            - Stopword removing
            - Removing of Special Charakters ('/,\,?')


        :param kwargs:
        :return: lst - preprocessed Text in from of ['one or two.' ,' Hello World.']
        '''

        print('Preprosses Text')

        df = pd.DataFrame(self.Corpus)

        # Bearbeiten der Leistungsplansystematik zu Verarbeitbaren String
        for String in df[0].values:
            s = String
            if type(s) == str:
                s = s.split(' [')
                s = s[0]
                s = s.split(' - ')
                s = s[0]
                df = df.replace({String: s})

        df[0] = df.fillna(' ')
        df_BackUp = df.copy()
        df = df[df[0] != ' ']

        Arr_KeyWord_df_pre = df[0].values.tolist()
        #print('Länge Corpus', len(Arr_KeyWord_df_pre))



        Arr_KeyWord_df_New = pre.PrePross(Arr_KeyWord_df_pre,
                                          _comma=_comma,
                                          Fuzzy=Fuzzy,
                                          FuzzyRank=FuzzyRank,
                                          _reversed=_reversed,
                                          Remove_specCar=Remove_specCar,
                                          IgnoreWord_list=self.word_exclusion_list)
        preprossList = Arr_KeyWord_df_New

        self.PreProssTxT = preprossList

        return preprossList


    def ExtractKeywords(self, Algo = 'Textrank'):
        '''
        Generate Keywords.
        Two Algorithms Available: Textrank and TFIDF

        :param Algo: str - Which Algo to choose. Standard is Textrank
        :return: lst - Keywords as List of Strings
        '''

        if self.PreProssTxT:
            corpus = self.PreProssTxT
        else:
            corpus = self.Corpus

        if Algo == 'Textrank':

            type = 'TextRank'

            kw = [getTextrankKeywords(entry, self.stop_words) for entry in corpus]
            print('Created TR Keywords...')


            def genKw(corpus):
                for x in range(len(corpus)):
                    projekt = []
                    for i in range(len(corpus[x])):
                        word = corpus[x][i][0]
                        projekt.append(word)
                    yield ','.join(map(str, projekt))


            kw_v2 = list(genKw(kw))

            #if saveFile:
            #    filename = 'KeywordArray_TextRank.sav'
            #    pickle.dump(kw_v2, open(filename, 'wb'))



        if Algo == 'TFIDF':

            type = 'TFIDF'

            tfdf = tfIdf_Vectorizer_train(corpus,stop_word = self.stop_words)

            kw = [tfIdf_Vectorizer_getKeyWords(tfdf, i, n=10).index.tolist() for i in range(0,len(corpus))]


            def genKw(corpus):
                for entry in corpus:
                    bagofwords = ','.join(map(str, entry))
                    yield bagofwords


            kw_v2 = list(genKw(kw))

        print('Keywords Created..')

        self.Keywords = kw_v2

        return kw_v2

    def CreatePairsFromKeywords(self, saveFile = False):
        '''

        Generate Keywords.
        Two Algorithms Available: Textrank and TFIDF

        :param saveFile: bol - Creates CSV from Edges
        :return: df - Creates Dataframe with Edges for a Graph
        '''
        kw_v2 = self.Keywords

        if kw_v2:

            from processor.graph_matrix_functions import Countmatrix, GetPairsNumpy, GetPairsWithWeight

            KeyM = Countmatrix(kw_v2, pandas=False, matrix=True)

            Edges = GetPairsNumpy(KeyM)


            #KeyM = pd.DataFrame(KeyM[0].todense(), columns=KeyM[1])
            Edges = pd.DataFrame(Edges, columns=['Source', 'Target', 'Projekt'])

            self.Edges = Edges

            if saveFile:
                    Edges.to_csv('KeywordsOutput\\Complete_edges.csv', sep=';', encoding='cp1252')

            print('Pairs Created..')

        else:

            print('No Keywords Found..')




    def CreateNetwork(self):
        '''
        Generate Gaph/Network in NetworkX Format.

        :return: G - Graph


        '''

        Edges = self.Edges



        if Edges.empty == False:
            '''Fals ein Network ausgegeben werden soll, network auf True stellen'''
            # Siehe https://www.kaggle.com/ferdzso/knowledge-graph-analysis-with-node2vec
            # siehe https://orbifold.net/default/node2vec-embedding/
            import networkx as nx

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

            self.Graph = G
            return G

        else:
            print('No Edges Found...')



    

