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
    """
    A custom transformer that counts occurrences of specific part-of-speech (POS) tags.

    This transformer takes a collection of text documents as input and outputs a numpy array
    where each row corresponds to a document, and each column represents the count of a specific
    POS tag within that document. Currently, it focuses on nouns (NN), verbs (VB), adjectives (JJ),
    and adverbs (RB), but this selection can be customized as needed.

    Methods:
    - fit(X, y=None): Placeholder method for compatibility with scikit-learn pipelines.
    - transform(X): Transforms the input documents into a feature matrix based on POS tag counts.

    Parameters:
    - X: A list or array-like object containing text documents to be processed.

    Returns:
    - A numpy array where each element i,j represents the count of a specific POS tag in document i.
    """
    def fit(self, X, y=None):
        # This transformer does not need to learn anything from the data,
        # so the fit method simply returns self.
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
    """
    A custom transformer that extracts sentiment features using TextBlob.

    This transformer computes the polarity and subjectivity of each text document
    provided as input and outputs a feature matrix where each row corresponds to a
    document, the first column is the polarity score, and the second column is the
    subjectivity score of the document.

    Polarity is a float within the range [-1.0, 1.0], where -1.0 indicates negative
    sentiment, 0 indicates neutral sentiment, and 1.0 indicates positive sentiment.
    Subjectivity is a float within the range [0.0, 1.0], where 0.0 is very objective
    and 1.0 is very subjective.

    Methods:
    - fit(X, y=None): Placeholder method for compatibility with scikit-learn pipelines, does nothing.
    - transform(X): Transforms the input documents into a feature matrix of sentiment scores.

    Parameters:
    - X: A list or array-like object containing text documents to be analyzed for sentiment.

    Returns:
    - A numpy array of shape (n_documents, 2), where n_documents is the number of documents
      in X. Each row in the array corresponds to a document, with the first column containing
      the polarity score and the second column containing the subjectivity score.
    """
    def fit(self, X, y=None):
        # This transformer does not need to learn anything from the data,
        # so the fit method simply returns self.
        return self
    
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

def load_data(database_filepath: str):
    """
    Load and preprocess data from a SQLite database for machine learning.

    This function reads the 'messages' table from the specified SQLite database,
    performs some preprocessing steps, and prepares the dataset for use in
    machine learning models. Specifically, it:
    - Replaces values of 2 with 1 in the 'related' column to handle a specific
      case in the dataset encoding.
    - Separates the dataset into features (X) and labels (Y).
    - Extracts the column names for the label columns.

    Parameters:
    - database_filepath (str): The file path to the SQLite database containing the 'messages' table.

    Returns:
    - X (numpy.ndarray): The feature array (messages) extracted from the 'message' column.
    - Y (numpy.ndarray): The label array, with each column corresponding to a different category
      label extracted from all columns except 'id', 'message', 'original', and 'genre'.
    - columns (numpy.ndarray): An array of the label column names, useful for identifying output
      from a machine learning model.
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("messages", engine)
    ## Replace value 2 with 1 in the 'related' column to normalize the label encoding
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    # Extract features and labels
    X = df["message"].values
    Y = df.drop(columns=["id", "message", "original", "genre"]).values
    # Extract label column names for later use in model output interpretation
    columns = df.iloc[:,4:].columns.values
    return X, Y, columns

def tokenize(text: str) -> list:
    """
    Process text data by normalizing, tokenizing, removing stopwords, and lemmatizing.

    The function performs the following operations on the input text:
    1. Converts the text to lowercase to normalize the case.
    2. Tokenizes the lowercase text into individual words.
    3. Removes stopwords (commonly used words that are unlikely to be useful for analysis)
       from the tokenized words.
    4. Lemmatizes the words, converting them to their base or dictionary form.

    Parameters:
    - text (str): The text to be processed.

    Returns:
    - list: A list of processed words after tokenization, stopword removal, and lemmatization.
    """
    text = text.lower()
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed

def build_model():
    """
    Build a machine learning pipeline for text classification.

    This function constructs a pipeline that processes text data through various
    transformations and uses a multi-output classifier for prediction. The pipeline
    includes:
    - Text feature extraction with TF-IDF,
    - Part-of-speech (POS) tagging counts,
    - Sentiment analysis features,
    and employs a RandomForestClassifier with balanced class weights for the final classification.

    The pipeline is structured as follows:
    1. A 'FeatureUnion' to combine features generated from:
       a. TF-IDF Vectorizer: Converts text into a matrix of TF-IDF features.
       b. POS Tag Counter: Counts specific POS tags within the text.
       c. TextBlob Sentiment Extractor: Extracts polarity and subjectivity scores from the text.
    2. A 'MultiOutputClassifier' that wraps a RandomForestClassifier, capable of handling
       multi-label classification tasks.

    Returns:
    - A constructed scikit-learn Pipeline object ready for fitting and making predictions.
    """

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
    """
    Evaluate the performance of a machine learning model on test data.

    This function predicts labels for the test set using the provided model and
    prints out the classification report for each output category. The classification
    report includes precision, recall, f1-score, and support for each label.

    Parameters:
    - model: The trained machine learning model to evaluate. This model should
      support the `predict` method.
    - X_test: The test set features (numpy array or pandas DataFrame) to use for
      making predictions.
    - Y_test: The true labels for the test set (numpy array or pandas DataFrame).
      Shape should be (n_samples, n_labels) where n_labels matches the number
      of categories.
    - category_names: List of strings representing the names of the output
      categories. This should match the columns in Y_test.

    Returns:
    - None. The function directly prints the classification report for each category.
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save a trained machine learning model to a file using pickle.

    Parameters:
    - model: The trained model to be saved.
    - model_filepath (str): Path to save the serialized model file.

    Returns:
    - None
    """
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