import pymysql
from pandas import DataFrame
import unicodedata

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

        sql_statement = 'SELECT PARAGRAPH FROM PARAGRAPHS WHERE POLARITY = "%s"' % label
        print(sql_statement)

        cursor.execute(sql_statement)
        print(cursor.rowcount, ' Paragraphs encountered')
        data = cursor.fetchall()
        db.close()

        return data

    def getDataTextAndLabel(self):
        db = pymysql.connect(self.server_address, self.username, self.password, self.target_database)
        cursor = db.cursor()

        sql_statement = 'SELECT PARAGRAPH, POLARITY FROM PARAGRAPHS WHERE POLARITY IS NOT NULL AND trim(POLARITY) <> ""'
        print(sql_statement)

        cursor.execute(sql_statement)
        print(cursor.rowcount, ' Paragraphs encountered')
        data = cursor.fetchall()
        db.close()

        return data


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

