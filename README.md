# Disaster-Response-Pipeline
Udacity Data Science Nanodegree Project- Disaster Response Pipeline
In this project, we analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

### Table of Contents

1. [Installation](#installation)
2. [File Descriptions](#files)
3. [Instructions](#Instructions)
3. [Results](#results)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*. Install all requirements from requirements.txt before running the code.


### Instructions: <a name="Instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
## File Descriptions <a name="files"></a>

There is two iPython notebooks available here for preparing ETL pipeline and ML pipeline and experimenting different models. The Data folder contains two csv files (messages and categories) used by this analysis.  



## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credits must be given to Figure Eight for the data and Udacity for the project idea.