import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
from textblob import TextBlob
import pickle
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class POSTagCounter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        pos_counts = []
        for document in X:
            # Perform minimal tokenization suitable for POS tagging
            tokens = word_tokenize(document)
            # Get POS tags for the tokenized text
            pos_tags = nltk.pos_tag(tokens)
            # Count occurrences of each POS tag
            tag_counts = Counter(tag for word, tag in pos_tags)
            # Extract counts for a few POS tags (customize as needed)
            pos_counts.append([tag_counts.get(tag, 0) for tag in ['NN', 'VB', 'JJ', 'RB']])
        return np.array(pos_counts)

class TextBlobSentimentExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # No fitting necessary
    
    def transform(self, X):
        # Initialize arrays to store the sentiment features
        polarity = np.zeros(len(X))
        subjectivity = np.zeros(len(X))
        
        # Extract sentiment for each document
        for i, text in enumerate(X):
            analysis = TextBlob(text)
            polarity[i] = analysis.sentiment.polarity
            subjectivity[i] = analysis.sentiment.subjectivity
        
        # Return the sentiment features as a feature matrix
        return np.column_stack([polarity, subjectivity])

def load_data(database_filepath):
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("messages", engine)
    #drop columns with 0 variance
    df = df.drop(df.std()[df.var() == 0].index.values, axis=1)
    #replace value 2 with 1 in related col
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    X = df["message"].values
    Y = df.drop(columns=["id", "message", "original", "genre"]).values
    columns = df.iloc[:,4:].columns.values
    return X, Y, columns

def tokenize(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed

def build_model():
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('tfidf', Pipeline([
            ('vec', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('pos_tags', POSTagCounter()),
        ('sentiment', TextBlobSentimentExtractor())
    ])),
    ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(class_weight="balanced")))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()