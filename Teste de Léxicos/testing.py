from utils import *
from sklearn.naive_bayes  import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import nltk

# Recuperação dos textos - treinar o modelo somente com as classes POS e NEG
all_data = get_data_from_db()
# all_data = all_data[(all_data["labels"] == "PO") | (all_data["labels"] == "NG")]
print('Textos carregados')

# Stopwords
stopwords = nltk.corpus.stopwords.words('portuguese')

# Léxico
subs_pos, subs_neg, verbs_pos, verbs_neg, adj_pos, adj_neg = load_claudia_freitas_lexicon()
lexicon = list(set(subs_neg + subs_pos + verbs_neg +verbs_pos + adj_neg + adj_pos))

# Features
features = FeatureUnion([
                    ("bigram", CountVectorizer(ngram_range=(1,2), stop_words= stopwords, binary= True)),
                    ("lexicon_vector", CountVectorizer(vocabulary= lexicon)),
                    ])

classifier, sKFold = run_cross_validation(all_data, features, MultinomialNB(), 10, True)
export_probabilities(classifier, all_data, sKFold)