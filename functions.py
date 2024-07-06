# Prepare libraries and data
import os
import pickle
import nltk
import re
import heapq
import pandas as pd
import matplotlib.pyplot as plt
from string import punctuation
from bs4 import BeautifulSoup
from urllib.request import urlopen
from nltk.stem.isri import ISRIStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score

# Text categories
categories = ['Economy & Business', 'Diverse News', 'Politic', 'Sport', 'Technology']

# Text Processor Class
class TextProcessor:
    def __init__(self, input_text, ar_text=False):
        self.input_text = input_text
        self.ar_text = ar_text

    def delete_links(self):
        pattern = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
        self.input_text = re.sub(pattern, ' ', self.input_text)

    def delete_repeated_characters(self):
        pattern = r'(.)\1{2,}'
        self.input_text = re.sub(pattern, r"\1\1", self.input_text)

    def replace_letters(self):
        replace = {"أ": "ا", "ة": "ه", "إ": "ا", "آ": "ا"}
        replace = dict((re.escape(k), v) for k, v in replace.items())
        pattern = re.compile("|".join(replace.keys()))
        self.input_text = pattern.sub(lambda m: replace[re.escape(m.group(0))], self.input_text)

    def clean_text(self):
        replace = r'[/(){}\[\]|@âÂ,;\?\'\"\*…؟–’،!&\+-:؛-]'
        self.input_text = re.sub(replace, " ", self.input_text)
        words = nltk.word_tokenize(self.input_text)
        words = [word for word in words if word.isalpha()]
        self.input_text = ' '.join(words)

    def remove_vowelization(self):
        vowelization = re.compile(""" [ًٌٍَُِّْـ]""", re.VERBOSE)
        self.input_text = re.sub(vowelization, '', self.input_text)

    def delete_stopwords(self):
        stop_words = set(nltk.corpus.stopwords.words("arabic") + nltk.corpus.stopwords.words("english"))
        tokenizer = nltk.tokenize.WhitespaceTokenizer()
        tokens = tokenizer.tokenize(self.input_text)
        wnl = nltk.WordNetLemmatizer()
        lemmatizedTokens = [wnl.lemmatize(t) for t in tokens]
        self.input_text = ' '.join([w for w in lemmatizedTokens if w not in stop_words])

    def stem_text(self):
        st = ISRIStemmer()
        tokenizer = nltk.tokenize.WhitespaceTokenizer()
        tokens = tokenizer.tokenize(self.input_text)
        self.input_text = ' '.join([st.stem(w) for w in tokens])

    def process_text(self):
        self.delete_links()
        self.delete_repeated_characters()
        self.clean_text()
        self.delete_stopwords()
        if self.ar_text:
            self.replace_letters()
            self.remove_vowelization()
            self.stem_text()
        else:
            self.input_text = self.input_text.lower()
        return self.input_text

# Summarizer Class
class Summarizer:
    def __init__(self, input_text):
        self.input_text = input_text

    def nltk_summarizer(self, number_of_sentences):
        stopWords = set(nltk.corpus.stopwords.words("arabic") + nltk.corpus.stopwords.words("english"))
        word_frequencies = {}
        for word in nltk.word_tokenize(self.input_text):
            if word not in stopWords and word not in punctuation:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        maximum_frequency = max(word_frequencies.values())

        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

        sentence_list = nltk.sent_tokenize(self.input_text)
        sentence_scores = {}
        for sent in sentence_list:
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) < 30:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]

        summary_sentences = heapq.nlargest(number_of_sentences, sentence_scores, key=sentence_scores.get)
        summary = ' '.join(summary_sentences)
        return summary

# Data Loader Class
class DataLoader:
    def __init__(self, en_path, ar_path):
        self.en_data = pd.read_csv(en_path)
        self.ar_data = pd.read_csv(ar_path)

    def clean_data(self):
        self.en_data = self.en_data.replace("entertainment", "diverse news")
        self.en_data = self.en_data.replace("business", "economy & business")

        self.ar_data = self.ar_data.replace("diverse", "diverse news")
        self.ar_data = self.ar_data.replace("culture", "diverse news")
        self.ar_data = self.ar_data.replace("politic", "politics")
        self.ar_data = self.ar_data.replace("technology", "tech")
        self.ar_data = self.ar_data.replace("economy", "economy & business")
        self.ar_data = self.ar_data.replace("internationalNews", "politics")
        self.ar_data = self.ar_data[~self.ar_data['type'].str.contains('localnews')]
        self.ar_data = self.ar_data[~self.ar_data['type'].str.contains('society')]

    def process_data(self):
        self.en_data['Processed Text'] = self.en_data['Text'].apply(lambda x: TextProcessor(x).process_text())
        self.ar_data['Processed Text'] = self.ar_data['text'].apply(lambda x: TextProcessor(x, ar_text=True).process_text())

    def encode_labels(self):
        en_label_encoder = LabelEncoder()
        self.en_data['Category Encoded'] = en_label_encoder.fit_transform(self.en_data['Category'])

        ar_label_encoder = LabelEncoder()
        self.ar_data['Category Encoded'] = ar_label_encoder.fit_transform(self.ar_data['type'])
        self.ar_data['Category Encoded'] = self.ar_data['Category Encoded'].replace(1, 0)
        self.ar_data['Category Encoded'] = self.ar_data['Category Encoded'].replace(0, 1)

    def split_data(self):
        en_X_train, en_X_test, en_y_train, en_y_test = train_test_split(self.en_data['Processed Text'], self.en_data['Category Encoded'], test_size=0.2, random_state=0)
        ar_X_train, ar_X_test, ar_y_train, ar_y_test = train_test_split(self.ar_data['Processed Text'], self.ar_data['Category Encoded'], test_size=0.2, random_state=0)
        return en_X_train, en_X_test, en_y_train, en_y_test, ar_X_train, ar_X_test, ar_y_train, ar_y_test

# Model Trainer Class
class ModelTrainer:
    def __init__(self, models, checkpoint_dir):
        self.models = models
        self.checkpoint_dir = checkpoint_dir

    def tfidf_features(self, X_train, X_test, ngram_range):
        tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, ngram_range))
        X_train = tfidf_vectorizer.fit_transform(X_train)
        X_test = tfidf_vectorizer.transform(X_test)
        return X_train, X_test, tfidf_vectorizer

    def save_checkpoint(self, model, vectorizer, model_name, dataset_name):
        model_path = os.path.join(self.checkpoint_dir, f'{model_name}_{dataset_name}_model.pkl')
        vectorizer_path = os.path.join(self.checkpoint_dir, f'{model_name}_{dataset_name}_vectorizer.pkl')

        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)

        with open(vectorizer_path, 'wb') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)

    def load_checkpoint(self, model_name, dataset_name):
        model_path = os.path.join(self.checkpoint_dir, f'{model_name}_{dataset_name}_model.pkl')
        vectorizer_path = os.path.join(self.checkpoint_dir, f'{model_name}_{dataset_name}_vectorizer.pkl')

        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)

            with open(vectorizer_path, 'rb') as vectorizer_file:
                vectorizer = pickle.load(vectorizer_file)

            return model, vectorizer
        else:
            return None, None

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, dataset_name):
        accuracies = {}

        for model_name, model in self.models.items():
            loaded_model, loaded_vectorizer = self.load_checkpoint(model_name, dataset_name)

            if loaded_model and loaded_vectorizer:
                X_train_vec = loaded_vectorizer.transform(X_train)
                X_test_vec = loaded_vectorizer.transform(X_test)
                accuracy = accuracy_score(y_test, loaded_model.predict(X_test_vec))
                print(f"Loaded {model_name} for {dataset_name} dataset. Accuracy: {accuracy:.4f}")
            else:
                X_train_vec, X_test_vec, tfidf_vectorizer = self.tfidf_features(X_train, X_test, 2)
                model.fit(X_train_vec, y_train)
                accuracy = accuracy_score(y_test, model.predict(X_test_vec))
                print(f"Trained {model_name} for {dataset_name} dataset. Accuracy: {accuracy:.4f}")
                self.save_checkpoint(model, tfidf_vectorizer, model_name, dataset_name)

            accuracies[model_name] = accuracy

        return accuracies

    def plot_accuracies(self, accuracies):
        model_names = list(accuracies.keys())
        accuracy_values = list(accuracies.values())

        plt.figure(figsize=(10, 5))
        plt.bar(model_names, accuracy_values, color='skyblue')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracies')
        plt.xticks(rotation=45)
        plt.show()

# Main function
def main():
    nltk.download('punkt')
    nltk.download('stopwords')

    en_path = 'data/bbc_news_dataset.csv'
    ar_path = 'data/arabic_dataset.csv'
    checkpoint_dir = 'models'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    data_loader = DataLoader(en_path, ar_path)
    data_loader.clean_data()
    data_loader.process_data()
    data_loader.encode_labels()

    en_X_train, en_X_test, en_y_train, en_y_test, ar_X_train, ar_X_test, ar_y_train, ar_y_test = data_loader.split_data()

    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier(),
        'LightGBM': LGBMClassifier(),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'MLP': MLPClassifier(max_iter=1000),
        'AdaBoost': AdaBoostClassifier(),
        'Extra Trees': ExtraTreesClassifier()
    }

    model_trainer = ModelTrainer(models, checkpoint_dir)
    en_accuracies = model_trainer.train_and_evaluate(en_X_train, en_X_test, en_y_train, en_y_test, 'English')
    ar_accuracies = model_trainer.train_and_evaluate(ar_X_train, ar_X_test, ar_y_train, ar_y_test, 'Arabic')

    print("English Dataset Accuracies:")
    print(en_accuracies)
    print("\nArabic Dataset Accuracies:")
    print(ar_accuracies)

    model_trainer.plot_accuracies(en_accuracies)
    model_trainer.plot_accuracies(ar_accuracies)

if __name__ == "__main__":
    main()
