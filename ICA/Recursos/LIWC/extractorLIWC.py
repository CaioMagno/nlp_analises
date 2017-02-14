
POS_EMOTION = '126'
NEG_EMOTION = '127'

f = open("tabela_palavras.txt",'r')
file_content = f.readlines()
f.close()

positive_lexicon = []
negative_lexicon = []

print('Procurando palavras')
for line in file_content:
    tokens = line.strip().split('\t')
    word = tokens[0]
    if POS_EMOTION in tokens:
        positive_lexicon.append(word)
    elif NEG_EMOTION in tokens:
        negative_lexicon.append(word)

print('escrevndo arquivo')
positive_file = open('positive_lexicon.txt','w')
negative_file = open('negative_lexicon.txt','w')

positive_file.writelines([word + '\n' for word in positive_lexicon])
negative_file.writelines([word + '\n' for word in negative_lexicon])

positive_file.close()
negative_file.close()

print('concluído')