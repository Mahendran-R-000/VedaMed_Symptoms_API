from bs4 import BeautifulSoup
from flask import Flask, request, json
import pickle
import pandas as pd
import requests
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
import nltk

# Download NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)

# Utilities for pre-processing
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')

# Load the model and data
model_tree = pickle.load(open("model_added.pkl", 'rb'))
df_norm_tree = pd.read_csv("dis_sym_dataset_norm.csv")
Y_tree = df_norm_tree.iloc[:, 0:1]
X_tree = df_norm_tree.iloc[:, 1:]
dataset_symptoms_tree = list(X_tree.columns)

def synonyms_tree(term):
    synonyms = []
    response = requests.get('https://www.thesaurus.com/browse/{}'.format(term))
    soup = BeautifulSoup(response.content,  "html.parser")
    try:
        container = soup.find('section', {'class': 'MainContentContainer'})
        row = container.find('div', {'class': 'css-191l5o0-ClassicContentCard'})
        row = row.find_all('li')
        for x in row:
            synonyms.append(x.get_text())
    except:
        None
    for syn in wordnet.synsets(term):
        synonyms += syn.lemma_names()
    return set(synonyms)

@app.route('/disease', methods=['POST'])
def classify_tree():
    symptoms_tree = str(request.form.get('syptoms')).lower().split(',')

    sample_x_tree = [0 for x in range(0, len(dataset_symptoms_tree))]
    for val in symptoms_tree:
        sample_x_tree[dataset_symptoms_tree.index(val)] = 1

    predicted_disease_tree = model_tree.predict_proba([sample_x_tree])
    k_tree = 3
    diseases_tree = list(set(Y_tree['label_dis']))
    diseases_tree.sort()
    topk_tree = predicted_disease_tree[0].argsort()[-k_tree:][::-1]

    topk_dict_tree = {}
    for idx, t in enumerate(topk_tree):
        match_sym_tree = set()
        row = df_norm_tree.loc[df_norm_tree['label_dis'] == diseases_tree[t]].values.tolist()
        row[0].pop(0)

        for idx, val in enumerate(row[0]):
            if val != 0:
                match_sym_tree.add(dataset_symptoms_tree[idx])
        prob = (len(match_sym_tree.intersection(set(symptoms_tree))) + 1) / (len(set(symptoms_tree)) + 1)
        topk_dict_tree[t] = prob

    j_tree = 0
    topk_index_mapping_tree = {}
    topk_sorted_tree = dict(sorted(topk_dict_tree.items(), key=lambda kv: kv[1], reverse=True))
    result_disease_tree =[]
    for key in topk_sorted_tree:
        prob = topk_sorted_tree[key]*100
        result_disease_tree.append(diseases_tree[key])
        topk_index_mapping_tree[j_tree] = key
        j_tree += 1

    result_tree = json.dumps({'result': result_disease_tree})
    return result_tree

@app.route('/EnterSymptoms', methods=['POST'])
def Enter_tree():
    Symptoms_tree = str(request.form.get('user_symtoms')).lower().split(',')

    user_symptoms_tree = []
    for user_sym in Symptoms_tree:
        user_sym = user_sym.split()
        str_sym_tree = set()
        for comb in range(1, len(user_sym)+1):
            for subset in combinations(user_sym, comb):
                subset = ' '.join(subset)
                subset = synonyms_tree(subset)
                str_sym_tree.update(subset)
        str_sym_tree.add(' '.join(user_sym))
        user_symptoms_tree.append(' '.join(str_sym_tree).replace('_', ' '))

    found_symptoms = set()
    for idx, data_sym in enumerate(dataset_symptoms_tree):
        data_sym_split = data_sym.split()
        for user_sym in user_symptoms_tree:
            count = 0
            for symp in data_sym_split:
                if symp in user_sym.split():
                    count += 1
            if count / len(data_sym_split) > 0.5:
                found_symptoms.add(data_sym)

    found_symptoms = list(found_symptoms)

    result = json.dumps({'result': found_symptoms})
    return result

if __name__ == '__main__':
    app.run()
