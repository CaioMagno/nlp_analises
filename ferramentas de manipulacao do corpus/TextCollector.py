from os import listdir
from os.path import isfile, join
from Newspaper import Newspaper
from TextCorpus import TextCorpus
from collections import defaultdict

import re
import pymysql
import xml.sax
import traceback
import SplittedText, RawText, Polarity
import encodings


def getAllXmlFiles(sourcePath):
    files = [f for f in listdir(sourcePath) if isfile(join(sourcePath, f)) if re.match('.*\.xml', f)]
    return files

def buildXmlParser():
    # Build a xml parser
    parser = xml.sax.make_parser()
    # turnoff the namespaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    return parser

def getAllXmlParsed(sourceDirectory, xmlFiles, classHandler):
    xmlParsedObjects = []
    xmlParser = buildXmlParser()

    for file in xmlFiles:
        handler = classHandler()
        xmlParser.setContentHandler(handler)
        xmlParser.parse(sourceDirectory + file)
        xmlParsedObjects.append(handler)

    return xmlParsedObjects

def getAllRawText():
    SOURCE_DIRECTORY = '/home/caiomagno/Documentos/Mestrado/Pesquisa/Recursos/corpus/C1/'
    xmlFiles = getAllXmlFiles(SOURCE_DIRECTORY)
    rawTextList = getAllXmlParsed(SOURCE_DIRECTORY, xmlFiles, RawText.RawText)
    return rawTextList

def getAllSplittedText():
    SOURCE_DIRECTORY = '/home/caiomagno/Documentos/Mestrado/Pesquisa/Recursos/corpus/paragraphs/'
    xmlFiles = getAllXmlFiles(SOURCE_DIRECTORY)
    splittedTextList = getAllXmlParsed(SOURCE_DIRECTORY, xmlFiles, SplittedText.SplittedText)
    return splittedTextList

def getAllPolarity():
    SOURCE_DIRECTORY = '/home/caiomagno/Documentos/Mestrado/Pesquisa/Recursos/corpus/polarity/'
    xmlFiles = [f for f in listdir(SOURCE_DIRECTORY) if isfile(join(SOURCE_DIRECTORY, f)) if re.match('.*M\.xml', f)]
    polarityList = getAllXmlParsed(SOURCE_DIRECTORY, xmlFiles, Polarity.Polarity)
    return polarityList

def gatherAllNews():
    polarities = getAllPolarity()
    splittedTexts = getAllSplittedText()
    rawTexts = getAllRawText()

    corpora = []

    for polarity in polarities:
        for splittedText in splittedTexts:
            if polarity.info['source'] == splittedText.info['id']:
                for rawText in rawTexts:
                    if splittedText.info['source'] == rawText.info['id']:
                        source = rawText.info['source']
                        break
                splittedTexts.remove(splittedText)
                break
        # corpus = buildTextCorpus(rawText, splittedText, polarity, Newspaper[source].value)
        corpus = TextCorpus(rawText, splittedText, polarity, Newspaper[source].value)
        corpora.append(corpus)
    return corpora

def splitNewsBySource(corpora):
    news_by_source = defaultdict(lambda: [])

    for corpus in corpora:
        news_by_source[corpus.source] += [(corpus.id_rawText, corpus.rawText)]

    return news_by_source

def buildTextCorpus(rawText, splittedText, polarity, source):
    text = (rawText.info['id'], rawText.text)
    paragraphs = splittedText.units
    corpus = TextCorpus(text, paragraphs, polarity, source)
    return corpus

def saveNewsInDatabase(corpora):
    db = pymysql.connect('localhost', 'root', '12345', 'CORPUS_VIES', charset = 'utf8')
    cursor = db.cursor()

    print('START INSERTING')
    for corpus in corpora:
        insertCorpusInDatabase(cursor, db, corpus)
        insertParagraphsInDatabase(cursor, db, corpus)

    print('DONE!')
    db.close()

def insertParagraphsInDatabase(cursor, db, corpus):
    SQL_INSERT_PARAGRAPH = 'INSERT INTO PARAGRAPHS(ID, ID_RAW_TEXT, TEXT_SOURCE, PARAGRAPH, ENTITY, POLARITY) \
        VALUES ("{}", "{}", "{}", "{}", "{}", "{}")'
    for paragraph in corpus.annotatedParagraphsTable.values():
        sql = SQL_INSERT_PARAGRAPH.format(*paragraph)
        execute_sql(cursor, db, sql)

def insertCorpusInDatabase(cursor, db, corpus):
    SQL_INSERT_NEWS = "INSERT INTO NEWS(ID_RAW_TEXT, TEXT_SOURCE, RAW_TEXT) VALUES (\'{}\', \'{}\', \'{}\')"
    sql = SQL_INSERT_NEWS.format(*(corpus.id_rawText, corpus.source, corpus.rawText))
    execute_sql(cursor, db, sql)


def execute_sql(cursor, db, sql):
    try:
        cursor.execute(sql)
        db.commit()
    except:
        print(sql)
        print(traceback.format_exc())
        db.rollback()
        raise Exception


if __name__ == '__main__':
    corpora = gatherAllNews()

    saveNewsInDatabase(corpora)
    pass