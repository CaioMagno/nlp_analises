import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.naive_bayes import MultinomialNB

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

def train_classifier(train, test, featurizer, classifier):
    pipeline = Pipeline([("features", featurizer), ("classifier", classifier)])
    pipeline.fit(train["texts"], train["labels"])
    predictions = pipeline.predict(test["texts"])

    accuracy = accuracy_score(test["labels"], predictions, normalize=True)
    return pipeline, accuracy

def run_cross_validation(all_data, features, classifier, n_folds=10, shuffle=True):
    sKFold = StratifiedKFold(n_splits= n_folds, shuffle= shuffle, random_state= True)

    classifier_models = []

    print("Cross Validation:")
    accuracy_average = np.array([])
    for index, (train, test) in enumerate(sKFold.split(all_data["texts"], all_data["labels"])):
        # Treinando um modelo Naive Bayes
        train_data = all_data.iloc[train]
        test_data = all_data.iloc[test]

        pipeline, accuracy = train_classifier(train_data, test_data, features, classifier)
        classifier_models.append(pipeline)

        accuracy_average = np.append(accuracy_average, accuracy)
        print("Fold ", index, " - Acuracia: ", accuracy)

    print("\nAccuracia media: ", accuracy_average.mean())
    print("Desvio padrão: ", accuracy_average.std())

    # Picking the best model
    best_classifier = classifier_models[accuracy_average.argmax(axis=0)]
    return best_classifier, sKFold

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