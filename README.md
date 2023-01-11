# dependency_parser
This project is implementation of the  Dependency Parsing
Using Bidirectional LSTM model from Kiperwasser and Goldberg 2016.
# The model
The model gives every pair of words in a sentence a weight from the first word to the second and the from the second to the first so for a sentence length n we have n^2 weights. The weights calculated in a way that the bigger the weight from a word to b word the more likely the dependecy between a to b. After it's finished attached weights the find the Maximum spaninng tree for all the edges between the words and that is the depandecy.
# Model Flow
Every sentence contain words and part of speech. First, all the sentences get embedding and after that we merge the word and it's pos embedding to one. From there every sentence is given as an input to bideractional LSTM with two layers. 
Then, for all the pair of words from the sentence we pass their output from the LSTM to a multi perceptron unit to get the weight of the edge beetween the first word to the second. The MLP unit return the weights in a n*n matrix which is given as input to Chu-Liu-Edmonds algorithem implementation that gives the MST that is translated to the dependecy.
# Model Architecture
Our model is has five parts:
The word embedding: Implemnted Word2Vec model that was train on the t
