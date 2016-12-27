import xml.sax
from xml.sax import ContentHandler

class RawText(ContentHandler):
    def __init__(self):
        self.currentData = ''
        self.currentID = ''
        self.info = {}
        self.text = ''

    def startElement(self, tag, attributes):
        self.currentData = tag

        if tag == 'info':
            type = attributes['type']
            value = attributes['value']
            self.info[type] = value

    def characters(self, content):
        if self.currentData == 'text':
            self.text = self.text + content

    def endElement(self, name):
        self.currentData = ''

    def __str__(self):
        text = 'News sample: id:{}, source:{} \nRawText:\n{}'.format(*(self.info['id'], self.info['source'], self.text))
        return text

if __name__ == '__main__':
    file = '/home/caiomagno/Documentos/Mestrado/Pesquisa/Recursos/corpus/C1/src_R001_C1_p2.xml'

    # Build a xml parser
    parser = xml.sax.make_parser()
    # turnoff the namespaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # Create an TextParagraph object that will stores the file content
    handler = RawText()
    parser.setContentHandler(handler)

    parser.parse(file)
    print(handler)