import nltk
import pymysql
import numpy as np
import machine_learning_utils
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from matplotlib import image
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from database_utils import DatabaseConnector, build_dataframe, normalize_text


def adjective_vectorizer(sentence, base_adjectives, form="vector"):
    normalized_sentence = normalize_text(sentence)
    normalized_sentence = sentence
    v = {adjective: normalized_sentence.split().count(adjective) for adjective in base_adjectives}
    if form == "dict":
        return v
    if form == "vector":
        return list(v.values())


def get_correlation_matrix(matrix_data):
    print("Tamanho do corpus: ", matrix_data.shape[0])
    print("Dimensionalidade: ", matrix_data.shape[1])

    corr_matrix = np.corrcoef(matrix_data)
    plt.figure()
    plt.imshow(np.multiply(corr_matrix, 255), cmap='Greys_r')
    plt.show()

    return corr_matrix


