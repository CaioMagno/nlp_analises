import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from database_utils import DatabaseConnector, build_dataframe, normalize_text
from sklearn.model_selection import StratifiedKFold

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

    return subs_pos, subs_neg, verbs_pos, verbs_neg, adj_pos, adj_neg


##########################################################################################################
######                   FUNÇÕES DE APRENDIZAGEM DE MÁQUINA                                       ########
##########################################################################################################

def run_classifier(train, test, featurizer, classifier):
    p = Pipeline([("features", featurizer), ("classifier", classifier)])
    p.fit(train["texts"], train["labels"])
    predictions = p.predict(test["texts"])

    accuracy = accuracy_score(test["labels"], predictions, normalize=True)
    return accuracy

def run_cross_validation(all_data, n_folds, shuffle, features, classifier):
    sKFold = StratifiedKFold(n_splits= n_folds, shuffle= shuffle, random_state= True)

    print("Cross Validation:")
    accuracy_average = np.array([])
    for index, (train, test) in enumerate(sKFold.split(all_data["texts"], all_data["labels"])):
        # Treinando um modelo Naive Bayes
        train_data = all_data.iloc[train]
        test_data = all_data.iloc[test]

        accuracy = run_classifier(train_data, test_data, features, classifier)
        accuracy_average = np.append(accuracy_average, accuracy)
        print("Fold ", index, " - Acuracia: ", accuracy)

    print("\nAccuracia media: ", accuracy_average.mean())
    print("Desvio padrão: ", accuracy_average.std())

def run_cross_validation2(X, Y, n_folds, shuffle, features, classifier):
    sKFold = StratifiedKFold(n_splits= n_folds, shuffle= shuffle, random_state= True)

    print("Cross Validation:")
    accuracy_average = np.array([])
    for index, (train, test) in enumerate(sKFold.split(X, Y)):
        # Treinando um modelo Naive Bayes
        train_data_X = X[train]
        train_data_Y = Y[train]
        train_data = {'texts': train_data_X, 'labels': train_data_Y}

        test_data_X = X.iloc[test]
        test_data_Y = Y.iloc[test]
        test_data = {'texts': test_data_X, 'labels': test_data_Y}

        accuracy = run_classifier(train_data, test_data, features, classifier)
        accuracy_average = np.append(accuracy_average, accuracy)
        print("Fold ", index, " - Acuracia: ", accuracy)

    print("\nAccuracia media: ", accuracy_average.mean())
    print("Desvio padrão: ", accuracy_average.std())

if __name__ == '__main__':
    all_data = get_data_from_db()
    all_data = all_data[(all_data["labels"] == "PO") | (all_data["labels"] == "NG")]
    print('Textos carregados')
    lexicon_CF = load_claudia_freitas_lexicon()
    features = FeatureUnion([("bigram", CountVectorizer(ngram_range=(1, 2), binary=True)),
                        ("lexicon_vector", CountVectorizer(vocabulary=lexicon_CF))])

    bag_of_features = features.fit_transform(all_data["texts"].toarray())
    pca = PCA(n_components=200)
    bag_of_features_reduced = pca.fit_transform(bag_of_features.toarray())
    run_cross_validation2(bag_of_features_reduced, all_data['labels'], 10, True, features,
                          SVC(C=316, kernel='sigmoid', coef0=0.5))