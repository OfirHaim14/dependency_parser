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
## The word embedding:
Implemnted Word2Vec model that was train on the train data words. The vector embedding size is 200.
## The pos embedding:
Implemnted Word2Vec model that was train on the train data pos. The vector embedding size is 40.
## LSTM:
Bidirectional LSTM with 2 layers. The input size is the size of the word and pos embedding togther which is 240. The hidden size is 300 and the output is 2*300=600 because the output is given for two words.
## MLP:
Implementation of the structure that was given in the article – MLP = w2*tanh(w1*x+b1)+b2, with simple neural network with two layers and bias. To get the score matrix we changed the output from the network to n on n matrix when so it could be given as a input to the chu liu Edmonds algorithm. In addition, to prevent overfitting the model have dropout layer between the first layer to the second after the tanh was activated.
## Chu Liu Edmonds
Implementation of the algo from the internet.

# Parameters:
Batch Size: the size the network gives is one but the loss calculates after every 15 sentences so the actual batch size is 15.  
Epochs – From trying different values after 25 there aren’t better results than for the first 25 and it’s just take more time to train.
Loss Functions – NLL loss
Optimizer – Adamax optimizer.
Drop out –Value that is too big run over too many neurons and value that is too small didn’t prevent the over fit so the best value after trying a lot of values the best results are between 0.25 and 0.35 the best is 0.3. 
