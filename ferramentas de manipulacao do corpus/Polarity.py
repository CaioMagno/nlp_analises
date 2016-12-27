import xml.sax
from xml.sax import ContentHandler

class Polarity(ContentHandler):
    def __init__(self):
        self.currentData = ''
        self.currentEntity = ''
        self.currentPolarity = ''
        self.currentID = ''

        self.info = {}
        self.units = {}

    def startElement(self, tag, attributes):
        self.currentData = tag
        if tag == 'info':
            type = attributes['type']
            value = attributes['value']
            self.info[type] = value
            if type == 'id' or type == 'source':
                self.info[type] = int(value)
        elif tag == 'unit': # Tanto unit quanto mark significam a mesma coisa: São os identificadores dos paragrafos no corpus paragraphs
            self.currentID = int(attributes['id'])
        elif tag == 'mark':
            self.currentID = int(attributes['unit'])
        elif tag == 'ann' and attributes['type'] == 'polarity':
            self.currentPolarity = attributes['value']
        elif tag == 'ann' and attributes['type'] == 'entity':
            self.currentEntity = attributes['value']
            self.units[self.currentID] = {'entity': self.currentEntity, 'polarity': self.currentPolarity}
            self.currentEntity = ''
            self.currentPolarity = ''
            self.currentID = ''
        else:
            pass



    # # Call when an elements ends
    # def endElement(self, tag):
    #     if self.currentData == 'info':
    #         print('INFO: ')
    #         print(self.info)
    #     elif self.currentData == 'unit':
    #         print('UNITS:')
    #         print(self.unit)
    #     self.currentData = ''

    def __str__(self):
        text = 'File Id' + str(self.info['id']) + '\n'
        for unitKey, unitValue in self.units.items():
            text += 'UNIT: ' + str(unitKey) + ' : ' + str(unitValue) + '\n'

        return text

def parse_xml(filename):
    # Build a xml parser
    parser = xml.sax.make_parser()
    # turnoff the namespaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # Create an TextParagraph object that will stores the file content
    handler = Polarity()
    parser.setContentHandler(handler)
    parser.parse(filename)
    print(handler)

if __name__ == '__main__':
    file = '/home/caiomagno/Documentos/Mestrado/Pesquisa/Recursos/corpus/polarity/ann_0001_polarity_paragraphs_0001_M.xml'
    parse_xml(file)