import json
import os
import sys
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from train_classifier import POSTagCounter, TextBlobSentimentExtractor



app = Flask(__name__)

def tokenize(text: str) -> list:
    """
    Tokenize, lemmatize, and clean text.

    This function processes the input text by performing several steps:
    1. Tokenizing the text into individual words,
    2. Lemmatizing each word to its base or dictionary form,
    3. Converting each word to lowercase, and
    4. Stripping whitespace from each word.

    The result is a list of clean, lemmatized tokens.

    Parameters:
    - text (str): The text to be tokenized and cleaned.

    Returns:
    - list: A list of clean, lemmatized tokens from the input text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disaster_response.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Render the main page of the web application with data visualizations.

    This function extracts necessary data for creating visualizations, prepares
    the visualizations using Plotly, and renders a web page with these visuals embedded.
    The visualizations include:
    - A bar chart showing the distribution of messages across different genres.
    - A bar chart showing the distribution of messages across various disaster categories,
      sorted by the count of messages in each category.

    The function uses data stored in a global DataFrame `df` which should contain
    message data with a 'genre' column and multiple category columns for classification.

    Returns:
    - A rendered HTML template ('master.html') with data visualizations embedded as
      Plotly graphs. The graphs are passed to the template as JSON objects.
    """
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_names = df.iloc[:, 4:].columns
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Disaster Message Types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
    ]
     
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Process user input and display classification results.

    This function handles the classification of user-provided text input. It retrieves the input via
    query parameters, uses a pre-trained machine learning model to classify the text, and then renders
    a web page displaying the classification results.

    The classification results are mapped to the respective categories defined in the 'df' DataFrame,
    starting from the fifth column onwards. The global 'model' variable is assumed to be the trained
    machine learning model capable of performing the prediction.

    Returns:
    - The rendered 'go.html' template with the original query and its classification results passed as
      context variables. 'query' contains the user-provided text, and 'classification_result' is a
      dictionary mapping category names to their predicted labels (0 or 1).

    Note:
    - The function assumes the existence of a global 'model' variable representing the trained
      classification model and a global 'df' DataFrame containing the category names. Ensure these
      are defined and accessible within the scope where this function is called.
    - The 'request' object is used to retrieve query parameters, which is available in Flask route
      functions. 'render_template' is part of Flask's templating engine, used to generate the final
      HTML response.
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()