#Import Python libraries

import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    '''
    Function to load 2 data set from the csv file and merge into a single data frame 
    return merged dataframe merged on common id column
    
    Input: messages_filepath, categories_filepath
    Output: Merged dataframe of messages and categories dataframe
    '''
    
    #Read csv's and load to dataframe variables
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #drop column orininal as it has more than 50% nan
    messages= messages.drop(['original'], axis = 1)
    #drop duplicate rows (all column same)
    messages = messages.drop_duplicates()
    messages = messages.drop_duplicates('id')
    
    #drop duplicate rows (all column same)
    categories = categories.drop_duplicates()
    categories = categories.drop_duplicates('id')
        
    
    # merge messages and categories dataframes on 'id' column
    df = pd.merge(left=messages, right=categories, how='left', left_on='id', right_on='id')
    
    return df
    
def clean_data(df):

    '''
    Function to clean data frame by 
    Splitting the values in the categories column on the ; character so that each value becomes a separate column
    Use the first row of categories dataframe to create column names for the categories data.
    Rename columns of categories with new column names
    Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
    Replace categories column in df with new category columns.
    
    Input: dataframe to be cleaned
    Output: cleaned dataframe
    '''
    
    # create a dataframe with 36 individual category columns
    categories = df['categories'].str.split(";",expand=True)
    
    #Use the first row of categories dataframe to create column names for the categories data.
    row = categories.iloc[0]
    category_colnames = row.str.split('-', expand = True)[0]
    
    #Rename columns of categories with new column names
    categories.columns = category_colnames
    
    #Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
    for column in categories:
        # set each value to be the last character of the string
        categories[column] =  categories[column].str.split('-').str.get(-1)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df = df.drop(labels="categories", axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df= df.drop_duplicates()
    
    # check for any null value in column message
    df['message'].isnull().values.any()
    
    # drop rows with NA in all column
    df.dropna(subset=['message'])
    
    
   
    return df
            

def save_data(df, database_filename):

    '''
    Function to store and save the cleaned dataframe into a sqllite database

    Input: df, database_filename
    Output: SQL Database 
    '''
    
    conn_str=f"sqlite:///{database_filename}"
    engine = create_engine(conn_str)
    df.to_sql('DisasterResponse', engine, index=False,if_exists = 'replace')
    

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