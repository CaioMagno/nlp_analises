import xml.sax
from xml.sax import ContentHandler

class SplittedText(ContentHandler):
    def __init__(self):
        self.currentData = ''
        self.currentID = ''
        self.info = {}
        self.units = {}

    # Call when an element starts
    def startElement(self, tag, attributes):
        self.currentData = tag
        if tag == 'info':
            type = attributes['type']
            value = attributes['value']
            self.info[type] = value
            if type == 'id':
                self.info[type] = int(value)
        elif tag == 'unit':
            id = attributes['id']
            self.currentID = id
            self.units[int(id)] = ''

    # Call when an elements ends
    def endElement(self, tag):
        # if self.currentData == 'info':
        #     print('INFO: ')
        #     print(self.info)
        # elif self.currentData == 'unit':
        #     print('UNITS:')
        #     print(self.unit)
        self.currentData = ''

    # Call when a character is read
    def characters(self, content):
        if self.currentData == 'unit':
            id = self.currentID
            self.units[int(id)] = content
            self.currentID = ''

    def __str__(self):
        text = 'Units found in file: \n'
        for key, value in self.units.items():
            text += '{} {} \n'.format(*(key, value))
        return text

if __name__ == '__main__':
    # Build a xml parser
    parser = xml.sax.make_parser()
    # turnoff the namespaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # Create an TextParagraph object that will stores the file content
    handler = SplittedText()
    parser.setContentHandler(handler)

    file = '/home/caiomagno/Documentos/Mestrado/Pesquisa/Recursos/corpus/gold_paragraphs/seg_0001_paragraphs_C1_R001_uam.xml'
    parser.parse(file)
    print(handler)
