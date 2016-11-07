from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.base import TransformerMixin
from sklearn import svm

import numpy
from pickle import load

from database_utils import DatabaseConnector, build_dataframe
from sklearn.naive_bayes import MultinomialNB

import nltk
from nltk.stem import RSLPStemmer, SnowballStemmer

class MLWrapper:
    def __init__(self, classifier):
        self.pipeline = Pipeline([
            # ('stemmer', Stemmer()),
            ('vectorizer', CountVectorizer()),
            # ('tfidf_transformer', TfidfTransformer()),
            ('classifier', classifier)
        ])

    def train(self, data):
        """

        :param data:  dataframe with columns text, label
        :return:
        """
        k_fold = KFold(n=len(data), n_folds=6)
        scores = []
        accuracies = []
        confusion = numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        for train_indices, test_indices in k_fold:
            stemmer = Stemmer()
            train_text = data.iloc[train_indices]['texts'].values
            train_text = stemmer.transform(train_text)

            vectorizer = CountVectorizer()
            feature_space = vectorizer.fit_transform(train_text)
            # print(feature_space.shape)

            train_y = data.iloc[train_indices]['labels'].values

            test_text = data.iloc[test_indices]['texts'].values
            test_text = stemmer.transform(test_text)
            test_y = data.iloc[test_indices]['labels'].values

            self.pipeline.fit(train_text, train_y)
            predictions = self.pipeline.predict(test_text)

            confusion += confusion_matrix(test_y, predictions, labels=["PO", "NG", "NE"])
            score = f1_score(test_y, predictions)
            accuracy = accuracy_score(test_y, predictions, normalize=True)

            scores.append(score)
            accuracies.append(accuracy)

        print('Total news classified:', len(data))
        print('Score:', sum(scores) / len(scores))
        print('Accuracy:', sum(accuracies) / len(accuracies))
        print('Confusion matrix:')
        print(confusion)

class Stemmer():
    def __init__(self):
        f = open('bigram_tagger.pkl','rb')
        self.tagger = load(f)
        f.close()
        pass

    def fit(self, X, y=None, **fit_params):
        return self

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

if __name__ == "__main__":
    print("Recuperando os textos")
    db_connector = DatabaseConnector('localhost', 'root', '12345', 'CORPUS_VIES')
    retrieved_data = build_dataframe(db_connector.getDataTextAndLabel())

    print("Treinando o modelo")
    ml_wrapper = MLWrapper(svm.SVC(C=316))
    # ml_wrapper = MLWrapper(MultinomialNB())
    ml_wrapper.train(retrieved_data)