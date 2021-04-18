# Disaster Response Pipeline Project

Classify Disaster Response Messages Classification of Disaster Response Messages using machine learning pipelines

# Table of Contents
  1. Installation
  2. Quick Start
  3. Project Overview
  4. File Descriptions
  5. Results
  6. Acknowledgements

## Installation

The code should run using Python versions 3.*. 

The necessary libraries are: 
  1. pandas 
  2. re 
  3. sys
  4. sklearn
  5. nltk
  6. sqlalchemy
  7. pickle

## Quick Start

Run the following commands in the project's root directory to set up your database and model.

1.  To run ETL pipeline that cleans data and stores in database 
    $ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
2.  To run ML pipeline that trains classifier and saves python
    $ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
3.  Run the following command in the app's directory to run your web app 
    $ python run.py
4.  Open another Terminal Window and type 
    $ env|grep WORK
5.  In a new web browser window, type in the following
    https://SPACEID-3001.SPACEDOMAIN  SPACEID and SPACEDOMAIN values would be known from above step 

## Project Overview

As part of this project , I have analyzed disaster data from source Figure Eight and built a model for an API that classifies disaster messages using a ML pipeline into 36 categories. categorized these events , so that any future incoming messages can be sent to an appropriate disaster relief agency. The dataset contains 26,248 pre-labelled text messages taken from real life disaster In the web app that is built with this training data one could enter any disaster related message and a classification to related categories will be given to the user as output. The model built will help to respond to future disaster events 
        
## File Descriptions

- Data
  - disaster_messages.csv: # CSV data file to process from figure8 for messages.
  - disaster_categories.csv: # Another CSV data file to process from figure8 for categories.
  - process_data.py: #Python script to clean and create a database.
  - DisasterResponse.db: # Database
- Model
  - train_classifier.py: # Python script of ML pipeline.
  - classifier.pkl: # Saved model by pickle library.
- App
  - run.py: # Python script or Flask file that runs app
  - template # folder
    - master.html # main page of web app
    - go.html # classification result page of web app
 -README.md

The file process_data.py contains an ETL pipeline that: 
  Loads data for messages and categories from csv
  Merges two datasets 
  Cleans the data 
  Saves it in a SQLite database DisasterResponse.db

The file train_classifier.py contains a NLP and ML pipeline that: 
  Loads data from database DisasterResponse.db 
  Splits data into training and test sets 
  Tokenize , Normalize text and build machine learning pipeline 
  Fit the model to Training set and tunes a model using GridSearchCV 
  Outputs results on the test set by using the best performance parameters from GridSearchCV 
  Stores the model in classifier.pkl so that it can be used by the Flask app

The file run.py contains a Flask web app that 
  Enables the user to enter a disaster message
  View the categories of the message. 
  Contains some visualizations that describe the data used to train the model.

## Results

Visuals

![Overview](https://github.com/pmbiradar/Disaster-Response-ML-Pipleline/blob/main/Overview%20of%20Training%20Set.PNG)

When a disaster message is submitted and the Classify Message button is clicked, the app shows how the message is classified by highlighting the categories in green.
![Message Classification](https://github.com/pmbiradar/Disaster-Response-ML-Pipleline/blob/main/Message%20Classification.PNG)



## Acknowledgements 

Udacity.com , Figure Eight
