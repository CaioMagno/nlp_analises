import re, string

class TextCorpus:
    def __init__(self, rawText, splittedText, polarity, source):
        """
        Esta classe agrupa todas as informações relevantes de uma notícia
        :param rawText: objeto RawText
        :param splittedText: objeto SplittedText
        :param polarity: objeto Polarity
        :param source: id do veiculo que divulgou a notícia
        """
        self.id_rawText = rawText.info['id']
        self.rawText = re.sub('[\'\"]', '', rawText.text)
        self.paragraphs = splittedText.units
        self.polarity = polarity.units
        self.source = source

        self.annotatedParagraphsTable = self.buildDatabaseModel()
        pass

    def buildDatabaseModel(self):
        annotated_paragraphs_table = {}
        p = '[' + string.punctuation + ']'
        for k in self.paragraphs.keys():
            paragraph_text = re.sub(p, ' ', self.paragraphs[k])
            paragraph_text = re.sub('"', '', paragraph_text)
            paragraph_text = re.sub("'", '', paragraph_text)
            annotated_paragraphs_table[k] = (k, self.id_rawText, self.source, paragraph_text, self.polarity[k]['entity'],
                                     self.polarity[k]['polarity'])

        return annotated_paragraphs_table