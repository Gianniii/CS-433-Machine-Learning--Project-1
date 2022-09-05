# Machine Learning Project 1
## Higgs Boson Challenge

Authors: Gianni Lodetti, Luca Bracone, Omid Karimi

In this project, the goal was to create a machine learning model for binary classification. The task was to detect Higgs Boson particles from background events using CERN’s LHC dataset. Some more information and dataset available on [AICrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs).

## Best Model
The code is available in the ```scripts/``` folder. <br />
To obtain the best result we obtained and reported on AICrowd, execute run.py file: ``` python run.py ```, <br /> the output file ```sub.csv``` is the best submission. <br />
<br />
Note: the datasets ```train.csv``` and ```test.csv``` have to be in the ```data/``` folder.

## Other Models and code
Most other models implemented and tested, with data exploration and data-preprocessing, can be found in the python notebook ```project1.ipynb```. <br />
<br />
The file ```implementations.py``` contains the six methods that had to be implemented. <br />
```helpers.py``` contains some helper functions to help for the implementations. <br />
```proj1_helpers.py``` contains given functions to help with data loading and label predictions.
