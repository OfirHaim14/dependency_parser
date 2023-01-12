import torch.nn as nn

ALPHA = 0.3  # 0.35 that changed with the layer size to 360 wasn't better


class Score(nn.Module):
    def __init__(self, lstm_output_size, hidden_mlp_dim, device):
        """
        Implementation to the score function that was given in the article,
        Score = MLP(W1, W2, b1, b2)(x) = W2*tanh(W1*x +b1) + b2.
        @param lstm_output_size: The size of output from the lstm(the size of the word's, and it's tag embedding)
        is the input to the score function, so it set the nn input size.
        @param hidden_mlp_dim: The hidden layer size - hyper parm
        """
        super(Score, self).__init__()
        self.lstm_output_size = lstm_output_size
        self.dropout = nn.Dropout(ALPHA)
        self.device = device
        self.hidden_MLP_dim = hidden_mlp_dim
        # we multiply by 2 because we get 2 words to calculate the edge between
        self.tanh = nn.Tanh()
        self.drop_out = nn.Dropout(ALPHA)
        self.w2 = nn.Linear(hidden_mlp_dim, 1, bias=True)
        self.first_w1 = nn.Linear(2 * lstm_output_size, hidden_mlp_dim, bias=True)
        self.second_w1 = nn.Linear(2 * lstm_output_size, hidden_mlp_dim)

    def forward(self, lstm_output, train=True):
        """
        Gets the embedding for the head, and it's tag, the modifier, and it's tag and calculate the score for the edge.
        @param lstm_output: The word embedding of the head and the modifier
        @param train: The dropout is used in the train part but not in the evaluation part, so this param
         Indicates whether we on the train part or evaluation part so the function will know if it should activate the
         dropout layer.
        @return: Score of the edge from head to modifier.
        """
        n = lstm_output.shape[0]
        # pass through the first layer of the perceptron
        first_linear_w1 = self.first_w1(lstm_output.to(self.device))
        m = first_linear_w1.shape[1]
        second_linear_w1 = self.second_w1(lstm_output.to(self.device))
        first_layer_matrix = first_linear_w1.unsqueeze(1) + second_linear_w1
        if train:  # we only use the dropout while training
            pre_scores_matrix = self.dropout(  # the dropout layer
                self.tanh(first_layer_matrix.transpose(0, 1).reshape(n ** 2, m).to(self.device)))  # make matrix
        else:
            pre_scores_matrix = self.tanh(first_layer_matrix.transpose(0, 1).reshape(n ** 2, m).to(self.device))
        scores_matrix = self.w2(pre_scores_matrix.to(self.device))  # pass through the second layer
        scores_matrix = scores_matrix.reshape((n, n))  # convert the shape to matrix
        scores_matrix = scores_matrix.transpose(0, 1)
        numpy_scores_matrix = scores_matrix.cpu().detach().numpy()
        return numpy_scores_matrix, scores_matrix
