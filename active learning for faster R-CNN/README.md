# 11785-project
To reimplement our experiment, use extract_labels to read label data from FLIR dataset first.  
Put all your images and the label file in datasets/yymnist folder in RovelMan's repo.  
Then replace config/test.yaml file in RovelMan's repo with test.yaml file in our repo.  
Run python learn.py configs/test.yaml  
Finally load your trained model and run visualize/visualize.py to visualize the model's predictions.  
