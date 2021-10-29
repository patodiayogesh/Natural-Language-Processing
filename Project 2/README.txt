Yogesh Patodiayp2607@columbia.edu

Problem Statement:
The data in data/crowdflower_data.csv consists of tweets 
labeled with 4 emotion la- bels: ‘neutral’, ‘happiness’, ‘worry’, and ‘sadness’. Each tweet has exactly one label.
We predict the label given a tweet

Task:
1) load, preprocess, and vectorize the data
2) load pre- trained 100-dimensional GloVe embeddings
3) train and test a Dense, RNN and CNN model
Description of Files:1. main.py: File Containig main() function. The file takes model name as input.   The choices of models are 'dense','RNN','CNN','extension'.   Contains implementation of train_model(), test_model() to train    and test data2. models.py: File Contains Implementation of Dense_Network, RNN and CNN   Each implementation trains the model based on their architecture3. utils.py: File Contains utility classes to create word embeddings, data cleaning and 
   implementation of Experimental_LR_Scheduler used for updating learning rate during training.Description of Models:1. DenseNetwork: Implements a Dense Network Architecture with 2 linear layers and one non-   linear activation function. F1 score = 0.442. RecurrentNetwork: Implements a GRU Recurrent Network with 2 hidden layer and a dense     linear layer. F1 score = 0.423. CNN: Implements a CNN Architecture. Works with various kernels to    convolute data and pass to linear and activation functions. F1 score = 0.49Best Model:CNN. It gives the highest F1 score.The file should be run from terminal with the following syntax:main.py --model <model>Please Note: Only one model runs at one time. 
Please download glove twitter embeddings and store in the file under resources