from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

from pickle import load
from nltk.stem import RSLPStemmer, SnowballStemmer
from sklearn.base import TransformerMixin, BaseEstimator
from database_utils import DatabaseConnector, build_dataframe
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import nltk
import numpy as np

class Stemmer(BaseEstimator, TransformerMixin):
    def __init__(self):
        f = open('bigram_tagger.pkl','rb')
        self.tagger = load(f)
        f.close()
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        stemmer = RSLPStemmer()

        listParagraphs = X.tolist()
        stemmed_paragraphs = []

        for paragraph in listParagraphs:
            words = paragraph.split()
            stemmed_words = []
            for word in words:
                tag = self.tagger.tag([word])
                if tag[0][1] == "NPROP":
                    continue
                stemmed_words.append(stemmer.stem(word))
            text = ''.join(w + ' ' for w in stemmed_words).strip()
            stemmed_paragraphs.append(text)

        return stemmed_paragraphs
    
    def transform(self, X, **fit_params):
        stemmer = RSLPStemmer()

        listParagraphs = X.tolist()
        stemmed_paragraphs = []

        for paragraph in listParagraphs:
            words = paragraph.split()
            stemmed_words = []
            for word in words:
                tag = self.tagger.tag([word])
                if tag[0][1] == "NPROP":
                    continue
                stemmed_words.append(stemmer.stem(word))
            text = ''.join(w + ' ' for w in stemmed_words).strip()
            stemmed_paragraphs.append(text)

        return stemmed_paragraphs

class Tagger():
    def __init__(self, filename):
        f = open(filename, 'rb')
        self.tagger = load(f)
        f.close()

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **fit_params):
        pass


def get_data_from_db(sentiment = None):
    db_connector = DatabaseConnector('localhost', 'root', '12345', 'CORPUS_VIES')
    if sentiment != None:
        retrieved_data = db_connector.getDataBySentiment(sentiment)
    else:
        retrieved_data = db_connector.getDataTextAndLabel()

    return retrieved_data

def load_tagger():
    f = open('bigram_tagger.pkl', 'rb')
    tagger = load(f)
    f.close()
    return tagger

def get_all_adjectives():
    corpus = get_data_from_db()
    corpus = corpus['texts'].tolist()
    tagger = load_tagger()

    adjectives = set()
    for text in corpus:
        adjectives_found = set([word for (word, tag) in tagger.tag(text.split()) if tag[:3] == 'ADJ'])
        adjectives = adjectives.union(adjectives_found)

    print("Numero de adjetivos encontrados: ", len(adjectives))
    return adjectives
# if __name__ == "__main__":
    # print("Recuperando os textos")
    # db_connector = DatabaseConnector('localhost', 'root', '12345', 'CORPUS_VIES')
    # retrieved_data = build_dataframe(db_connector.getDataTextAndLabel())
    #
    # print("Treinando o modelo")
    # ml_wrapper = MLWrapper(svm.SVC(C=316))
    # # ml_wrapper = MLWrapper(MultinomialNB())
    # ml_wrapper.train(retrieved_data)