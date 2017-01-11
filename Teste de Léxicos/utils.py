import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from database_utils import DatabaseConnector, build_dataframe, normalize_text
from sklearn.model_selection import StratifiedKFold

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