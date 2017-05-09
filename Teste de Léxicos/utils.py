import numpy as np
import pandas as pd
import re

from pickle import load
from pandas import DataFrame

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from nltk.stem import RSLPStemmer, SnowballStemmer
from sklearn.base import TransformerMixin, BaseEstimator
from database_utils import DatabaseConnector, build_dataframe, normalize_text
from sklearn.model_selection import StratifiedKFold

from nltk.corpus import stopwords

from sklearn.decomposition import PCA
from sklearn.svm import SVC

##########################################################################################################
######                   FUNÇÕES DE OBTENÇÃO DE DADOS                                             ########
##########################################################################################################

def get_data_from_db(sentiment = None):
    db_connector = DatabaseConnector('localhost', 'root', '12345', 'CORPUS_VIES')
    if sentiment != None:
        retrieved_data = db_connector.getDataBySentiment(sentiment)
    else:
        retrieved_data = db_connector.getDataTextAndLabel()

    return retrieved_data

def load_claudia_freitas_lexicon():
    def extract_words(filename):
        import re

        f = open(filename, 'r')
        file_content = ''.join(f.readlines())
        f.close()

        words = re.findall(r'\[(\w+)\]', file_content)
        return words

    subs_pos = extract_words('Recursos/Claudia Freitas/subs_pos')
    subs_neg = extract_words('Recursos/Claudia Freitas/subs_neg')
    verbs_pos = extract_words('Recursos/Claudia Freitas/verbs_pos')
    verbs_neg = extract_words('Recursos/Claudia Freitas/verbs_neg')
    adj_pos = extract_words('Recursos/Claudia Freitas/adj_pos')
    adj_neg = extract_words('Recursos/Claudia Freitas/adj_neg')
    
    # df = DataFrame(data = {"subs": [len(subs_pos), len(subs_neg)], "verbs": [len(verbs_pos), len(verbs_neg)], "adj": [len(adj_pos), len(adj_neg)]})

    return list(set(subs_pos + subs_neg + verbs_pos + verbs_neg + adj_pos + adj_neg))
    # return df

# Carregar léxico LIWC
# OBS: PALAVRAS STEMMIZADAS!
def get_LIWC_lexicon():
    def clean_words(word_list):
        words = [word.strip("\n") for word in word_list]
        words = [word.strip("*") for word in words]
        return words
    lex = clean_words(open("Recursos/LIWC/positive_lexicon.txt").readlines())
    lex = lex + clean_words(open("Recursos/LIWC/negative_lexicon.txt").readlines())
    return list(set(lex))

##########################################################################################################
######                   FUNÇÕES DE PRE PROCESSAMENTO                                             ########
##########################################################################################################


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


# O FIT_TRANSFORM DEVE RETORNAR UM DATAFRAME
class NumRemover(BaseEstimator, TransformerMixin):
    def fit_transform(self, X, y=None, **fit_params):
        # X_copy = X.copy()
        for line in X.iterrows():
            index = line[0]
            paragraph = line[1]["texts"]
            filtered_text = (re.sub('\d', ' NUM ', paragraph))
            X = X.set_value(index,"texts", filtered_text)
        return X

class WordRemover(BaseEstimator, TransformerMixin):
    def __init__(self, words_to_remove):
        self.words = words_to_remove

    def fit_transform(self, X, y=None, **fit_params):
        return self.remove_words(X, self.words)

    def remove_words(self, data_frame, word_list):
        for line in data_frame.iterrows():
            index = line[0]
            paragraph = line[1]["texts"]
            tokens = paragraph.split()
            tokens = [token for token in tokens if token not in self.words ]
            data_frame.ix[index, "text"] = ' '.join(tokens)
        return data_frame

##########################################################################################################
######                   FUNÇÕES DE APRENDIZAGEM DE MÁQUINA                                       ########
##########################################################################################################

def train_classifier(train, test, featurizer, classifier):
    pipeline = Pipeline([("features", featurizer), ("classifier", classifier)])
    pipeline.fit(train["texts"], train["labels"])
    predictions = pipeline.predict(test["texts"])

    accuracy = accuracy_score(test["labels"], predictions, normalize=True)
    return pipeline, accuracy

def run_cross_validation(all_data, features, classifier, n_folds=10, shuffle=True):
    sKFold = StratifiedKFold(n_splits= n_folds, shuffle= shuffle, random_state= True)

    # classifier_models = []

    print("Cross Validation:")
    accuracy_average = np.array([])
    for index, (train, test) in enumerate(sKFold.split(all_data["texts"], all_data["labels"])):
        # Treinando um modelo Naive Bayes
        train_data = all_data.iloc[train]
        test_data = all_data.iloc[test]

        pipeline, accuracy = train_classifier(train_data, test_data, features, classifier)
        # classifier_models.append(pipeline)

        accuracy_average = np.append(accuracy_average, accuracy)
        #print("Fold ", index, " - Acuracia: ", accuracy)

    print("Accuracia media: ", accuracy_average.mean())
    print("Desvio padrão: ", accuracy_average.std())

    # Picking the best model
    # best_classifier = classifier_models[accuracy_average.argmax(axis=0)]
    # return best_classifier, sKFold

def run_cross_validation2(data, labels, classifier, n_folds=10, shuffle=True):
    sKFold = StratifiedKFold(n_splits= n_folds, shuffle= shuffle, random_state= True)

    # classifier_models = []

    print("Cross Validation:")
    accuracy_average = np.array([])
    for index, (train, test) in enumerate(sKFold.split(data, labels)):
        # Treinando um modelo Naive Bayes
        train_data = data[train]
        test_data = data[test]

        pipeline, accuracy = train_classifier(train_data, test_data, features, classifier)
        # classifier_models.append(pipeline)

        accuracy_average = np.append(accuracy_average, accuracy)
        #print("Fold ", index, " - Acuracia: ", accuracy)

    print("Accuracia media: ", accuracy_average.mean())
    print("Desvio padrão: ", accuracy_average.std())
    
def evaluate(data, vectorizer, n_folds):
    print("Naive Bayes---------------------------------")
    run_cross_validation(data, vectorizer, MultinomialNB(), n_folds = n_folds)
    print("\nMaxEnt--------------------------------------")
    run_cross_validation(data, vectorizer, LogisticRegressionCV(fit_intercept=False, penalty= 'l2', dual= False), n_folds = n_folds)
    print("\nSVM-----------------------------------------")
    run_cross_validation(data, vectorizer, SVC(C=316), n_folds = n_folds)

def export_probabilities(classifier, all_data, sKFold):
    labels = all_data["labels"]
    predictions = classifier.predict_proba(all_data["texts"])

    classes = classifier.classes_
    print(classes)

    df_content ={'labels': labels}
    for index, classe in enumerate(classes):
        df_content[classe] = predictions[:,index]

    performance_report = DataFrame(df_content)

    writer = pd.ExcelWriter("report.xls", engine = "xlsxwriter")
    performance_report.to_excel(writer, "Sheet1")
    writer.save()
    print("Arquivo exportado")

if __name__ == '__main__':
    wr