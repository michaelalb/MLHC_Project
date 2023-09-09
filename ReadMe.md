# MLHC Project code file and folder descriptions

## code files:
All runnable files can be run on their own using their main or as a part of the flow in the jupyter notebook.
1. Consts.py: contains all the constants used in the project
2. DataExploration.py: contains the code for data exploration 
This contains all functions for data exploration we found insightful, of course there were ad-hoc exploration done which did we
did not find interesting enough to document.
There is no orchestrator function here, each analysis is done on its own.
3. DataLoader.py: contains the code for loading the data from the Physionet formats into both parquet files and 
mostly into Darts time series objects. Also train test splits, normalization and preparation for model training.  
4. ModelEvaluator.py: Contains all functions used for the  evaluation of the model using the official Physionet code.
5. ModelTraining.py: Contains the training loop used to run the model.
6. OfficialEvaluations.py: Untouched downloaded code from the Physionet official challenge website.
7. ResultsEvaluator.py: contains all the code for the evaluation of the results of the model - looking for biases, 
analyzing the feature importance and attention.
8. FullRun.ipynb: This contains a streamlined version of the code we used.  
This is what we run on our labs clusters to get all the results in the paper and should be used to get 
and overview of our project and our work.
I would not recommend actually running the model training. it takes a few days on a 4 core GPU.
