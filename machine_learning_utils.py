from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import numpy

from database_utils import DatabaseConnector, build_dataframe
import nltk

class MLWrapper:
    def __init__(self):
        self.pipeline = Pipeline([
            ('vectorizer', CountVectorizer(stop_words=nltk.corpus.stopwords.words('portuguese'))),
            # ('tfidf_transformer', TfidfTransformer()),
            ('classifier', MultinomialNB())
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
            train_text = data.iloc[train_indices]['texts'].values
            train_y = data.iloc[train_indices]['labels'].values

            test_text = data.iloc[test_indices]['texts'].values
            test_y = data.iloc[test_indices]['labels'].values

            self.pipeline.fit(train_text, train_y)
            predictions = self.pipeline.predict(test_text)

            confusion += confusion_matrix(test_y, predictions, labels=["PO", "NG", "NE"])
            score = f1_score(test_y, predictions)
            accuracy = accuracy_score(test_y, predictions, normalize=True)

            scores.append(score)
            accuracies.append(accuracy)

        print('Total emails classified:', len(data))
        print('Score:', sum(scores) / len(scores))
        print('Accuracy:', sum(accuracies) / len(accuracies))
        print('Confusion matrix:')
        print(confusion)




if __name__ == "__main__":
    print("Recuperando os textos")
    db_connector = DatabaseConnector('localhost', 'root', '12345', 'CORPUS_VIES')
    retrieved_data = build_dataframe(db_connector.getDataTextAndLabel())

    print("Treinando o modelo")
    ml_wrapper = MLWrapper()
    ml_wrapper.train(retrieved_data)