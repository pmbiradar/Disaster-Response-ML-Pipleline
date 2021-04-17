import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')


import pandas as pd
from sqlalchemy import create_engine

import re


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import train_test_split , GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report, f1_score, make_scorer, fbeta_score
import pickle

def load_data(database_filepath):
    '''
    Function to load  data from database to data frame 
        
    Input: database_filepath
    Output: Value from column message in variable 'X' , categories values in variable 'Y' and 
            list of all category column names in variable 'category_names'
    '''
    
    #Read database table and load to dataframe variables
    conn_str=f"sqlite:///{database_filepath}"
    engine = create_engine(conn_str)
    df = pd.read_sql_table('DisasterResponse', con=engine)
    
    #store rows only with value in column related 0 or 1
    df=df[(df['related'] == 0) | (df['related'] == 1)]
    
    #store value from column message , category column values and column names in Variable
    X = df['message']
    Y = df.iloc[:,3:]
    category_names = Y.columns.tolist()
    return X,Y,category_names  

def tokenize(text):

    '''
     function to split text based on white space and punctuation using  word_tokenize and
     normalize words using WordNetLemmatizer
    '''
    # changing all upper case character to lower case
    text = text.lower()
    # Removing any special character
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text) 
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens   
    
   

def build_model():

    '''Function to build pipeline, that should take the text and tokenize the
    input and output classification results on the other 36 categories
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    pipeline.get_params()
    
    #found below parameters as the best performance parameters with below code
    #parameters = {'tfidf__norm': ['l1','l2'],'clf__estimator__criterion': ["gini", "entropy"]}
    #cv = GridSearchCV(pipeline, param_grid=parameters)
    #cv.fit(X_train, Y_train)
    #cv.best_params_
    #output - {'clf__estimator__criterion': 'gini', 'tfidf__norm': 'l2'}

    best_parameters = {'tfidf__norm': ['l2'],'clf__estimator__criterion': ["gini"]}
    
    #perform Grid Search to find the bext parameters for the model
      
    model = GridSearchCV(pipeline, param_grid=best_parameters)
       
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Test model using sklearn's `classification_report`on each category columns
    '''
    Y_pred = model.predict(X_test)
    
    #Show the accuracy, precision, and recall of the tuned model.
    print(classification_report(Y_test, Y_pred,target_names=category_names))
        

def save_model(model, model_filepath):
    '''
    Exporting your model as a pickle file
    '''
    with open(model_filepath,'wb') as file:
        pickle.dump(model,file)
  


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