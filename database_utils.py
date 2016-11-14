import pymysql
import unicodedata

from pickle import load
from nltk import FreqDist
from pandas import DataFrame

class DatabaseConnector:
    def __init__(self, server_address, username, password, target_database):
        self.server_address = server_address
        self.username = username
        self.password = password
        self.target_database = target_database

    def getDataBySentiment(self, label):
        """
        :param label: It can be just one of these three labels: PO, NG, NE
        :return:
        """
        db = pymysql.connect(self.server_address, self.username, self.password, self.target_database)
        cursor = db.cursor()

        sql_statement = 'SELECT PARAGRAPH, POLARITY FROM PARAGRAPHS WHERE POLARITY = "%s"' % label        
        print(sql_statement)

        cursor.execute(sql_statement)
        print(cursor.rowcount, ' Paragraphs encountered')
        data = cursor.fetchall()
        db.close()

        data = list(data)
        return build_dataframe(data)

    def getDataTextAndLabel(self):
        db = pymysql.connect(self.server_address, self.username, self.password, self.target_database)
        cursor = db.cursor()

        sql_statement = 'SELECT PARAGRAPH, POLARITY FROM PARAGRAPHS WHERE POLARITY IS NOT NULL AND trim(POLARITY) <> ""'
        print(sql_statement)

        cursor.execute(sql_statement)
        print(cursor.rowcount, ' Paragraphs encountered')
        data = cursor.fetchall()
        db.close()

        return build_dataframe(data)

    def get_data_from_db(self, sentiment=None):
        db_connector = DatabaseConnector('localhost', 'root', '12345', 'CORPUS_VIES')
        if sentiment != None:
            retrieved_data = db_connector.getDataBySentiment(sentiment)
        else:
            retrieved_data = db_connector.getDataTextAndLabel()

        return retrieved_data

    def load_tagger(self):
        f = open('bigram_tagger.pkl', 'rb')
        tagger = load(f)
        f.close()
        return tagger

    def get_all_adjectives(self):
        corpus = self.get_data_from_db()
        corpus = corpus['texts'].tolist()
        tagger = self.load_tagger()

        adjectives = set()
        for text in corpus:
            adjectives_found = set([word for (word, tag) in tagger.tag(text.split()) if tag[:3] == 'ADJ'])
            adjectives = adjectives.union(adjectives_found)

        print("Numero de adjetivos encontrados: ", len(adjectives))
        return adjectives

    def get_adjective_by_sentiment(self, sentiment):
        lista = self.get_data_from_db(sentiment=sentiment)
        tagger = self.load_tagger()
        result_list = []
        for sentence in lista:
            result = tagger.tag(sentence[0].split())
            result_list += result

        fd = FreqDist([word for (word, tag) in result_list if tag[:3] == 'ADJ'])
        adj_set = set(fd.keys())
        print(len(adj_set), ' Adjectives encountered\n')

        return adj_set


def build_dataframe(dataset):
    """
    :param dataset: list of tuples (text, label)
    :return: dataframe with columns text, label
    """
    text_sequence = []
    label_sequence = []

    for text, label in dataset:
        text_sequence.append(normalize_text(text))
        label_sequence.append(label)

    dataframe = DataFrame({'texts': text_sequence, 'labels': label_sequence})
    return dataframe

def normalize_text(text):
    normalized_text = text.lower()
    # normalized_text = ''.join((c for c in unicodedata.normalize('NFD', normalized_text) if unicodedata.category(c) != 'Mn'))
    return normalized_text


if __name__ == "__main__":
    db_connector = DatabaseConnector('localhost', 'root', '12345', 'CORPUS_VIES')
    retrieved_data = build_dataframe(db_connector.getDataTextAndLabel())
    retrieved_data

