# Dependency Parser
This project is an implementation of the  Dependency Parsing
Using Bidirectional LSTM model from Kiperwasser and Goldberg 2016.
## The model
The model gives every pair of words in a sentence two weights, one from the first word in the pair to the second word and the other from the second word to the first word. So for a sentence length n we have n^2 weights because in dependency parsing a word can't connect to itself but there is a root token at the beginning of the sentence. The weights are calculated in a way that the bigger the weight from a word A to another word B, the more likely the dependency connection between A and B. After the model finishes attaching weights to every pair of words in the sentence the weights of all the pairs of words are given as input to Chu Liu Edmonds algorithm in order to find the Maximum Spanning Graph which in this case is the dependency parser. The Graph that the algorithm finds its M.S.G is a full graph from all the words in the sentences and the edges' weight is the one the model previously calculated. The M.S.G we get is the dependency of the sentence.
## Model Flow
Every sentence contains words and parts of speech. First, all the sentences get embedding and after that, we merge the word and its pos embedding into one. From there every sentence is given as an input to bidirectional LSTM with two layers. 
Then, for all the pairs of words from the sentence we pass their output from the LSTM to a multi-perceptron unit to get the weight of the edge between the first word to the second. The MLP unit returns the weights in a n*n matrix which is given as input to the Chu-Liu-Edmonds algorithm implementation that gives the MST that is translated to the dependency.
## Model Architecture
Our model has five units:
### The word embedding:
Implemented Word2Vec model that was trained on the train data words. The vector embedding size is 200.
### The pos embedding:
Implemented Word2Vec model that was trained on the train data pos. The vector embedding size is 40.
### LSTM:
Bidirectional LSTM with 2 layers. The input size is the size of a word's and pos' embedding combined which is 240. The hidden size is 300 and the output is 2*300=600 because we have two outputs of 300 one for each direction.
### MLP:
Implementation of the structure that was given in the article – MLP = w2 * tanh(w * x + b1) + b2, by using a simple neural network with two layers and bias. To get the score matrix the output from the network shape is converted to n on n matrix so it could be given as an input to the Chu Liu Edmonds algorithm. In addition, to prevent overfitting this MLP model has a dropout layer between the first layer to the second after the tanh is activated.
### Chu Liu Edmonds
Implementation of the algorithm from the internet.
## Parameters:
Batch Size: The size the network gives is one but the loss calculates after every 15 sentences so the actual batch size is 15.  
Epochs – From trying different values after 25 there aren’t better results than for the first 25 and it just takes more time to train.  
Loss Functions – NLL loss  
Optimizer – Adamax optimizer.  
Drop out –A dropout value that is too big ran over too many neurons and a value that is too small didn’t prevent the overfit so the after trying a lot of values the best results are between 0.25 and 0.35 the best is 0.3. 
![Uas on the train data](https://user-images.githubusercontent.com/118376368/212489020-b5909348-236e-466e-b6a7-0b9b56e0fcc2.png)
![Loss on the train data](https://user-images.githubusercontent.com/118376368/212489203-2834dbc4-d842-4db1-bc85-1ccfc7185675.png)
