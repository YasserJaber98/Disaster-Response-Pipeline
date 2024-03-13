import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Load and merge messages and categories datasets.

    This function reads messages and categories from their respective file paths, 
    merges them into a single DataFrame, and returns the merged DataFrame.

    Parameters:
    - messages_filepath (str): The file path for the messages dataset.
    - categories_filepath (str): The file path for the categories dataset.

    Returns:
    - DataFrame: A pandas DataFrame containing the merged content of messages and categories datasets.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data in the provided DataFrame.

    This function performs several cleaning steps on the input DataFrame:
    - Splits the 'categories' column into separate, clearly named columns,
    - Converts category values to binary (0 or 1),
    - Drops the original 'categories' column,
    - Concatenates the original DataFrame with the new category columns,
    - Removes duplicate rows.

    The 'categories' column is expected to contain semicolon-delimited strings indicating the category
    and a binary value (e.g., "related-1;request-0;offer-0;..."). This function splits these strings into
    separate columns, uses the category part as new column names, and the binary part as values in these columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame to clean, which must include a 'categories' column with semicolon-delimited category-value pairs.

    Returns:
    - pd.DataFrame: A DataFrame with the original data and added binary category columns, with duplicates removed.
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda e: e[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df = df.drop(columns=["categories"])
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    return df

def save_data(df: pd.DataFrame, database_filename: str):
    """
    Save the given DataFrame to a SQLite database.

    This function saves the provided DataFrame to a SQLite database file specified by
    the database_filename parameter. The data is saved in a table named 'messages'.
    If the table already exists in the database, it will be replaced.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be saved to the database.
    - database_filename (str): The filename (or path) for the SQLite database file where
      the DataFrame will be saved. If only a filename is provided, the database will
      be created in the current working directory; otherwise, it will be created/updated
      at the specified path.

    Returns:
    - None

    Note:
    - This function does not return a value but saves the DataFrame to the specified
      SQLite database file. Ensure that the provided DataFrame has the appropriate
      structure expected by downstream processes or database schema requirements.
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()