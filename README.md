# Disaster Response Project

## Project Overview
The Disaster Response Project utilizes machine learning to classify disaster messages into categories, enabling quicker and more efficient response from relief agencies. This project includes a web app where an emergency worker can input a new message and receive classification results in several categories.

## Installation
This project requires Python 3 and the following Python libraries installed:
- NumPy
- Pandas
- Matplotlib
- Json
- Plotly
- Nltk
- Flask
- Sklearn
- Sqlalchemy
- Sqlite3
- Textblob

## File Descriptions
- `data/process_data.py`: The ETL pipeline used for data cleaning, feature extraction, and storing data in a SQLite database.
- `models/train_classifier.py`: The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle.
- `app/run.py`: The Flask file to run the web application.

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run the ETL pipeline that cleans data and stores it in a database:
        ```
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db
        ```
    - To run the ML pipeline that trains the classifier and saves it:
        ```
        python models/train_classifier.py data/disaster_response.db models/model.pkl
        ```

2. Run the following command in the `app` directory to run the web app:
    ```
    python run.py
    ```

3. Go to http://0.0.0.0:3000/

## Web App Screenshots
![image](https://github.com/YasserJaber98/Disaster-Response-Pipeline/assets/65098704/a8c10e06-9b39-4d6d-9999-eb854fd911b0)
![image](https://github.com/YasserJaber98/Disaster-Response-Pipeline/assets/65098704/0d50197b-b489-49c6-8bee-76a6c067053e)
