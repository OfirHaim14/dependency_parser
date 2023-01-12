import torch.nn as nn
import torch
from chu_liu_edmonds import decode_mst
from Score import Score


def calc_loss(true, pred):
    """
    Calculate the loss between our dependency prediction to the true prediction.
    @param true: The true dependency.
    @param pred: Our prediction.
    :return: The loss we got
    """
    pred = torch.exp(pred)
    loss = 0
    for i in range(1, len(true)):
        loss += -torch.sum(torch.log(pred[true[i], i] / torch.sum(pred[:, i])))
    return loss / (len(true) - 1)


class DependencyParser(nn.Module):
    def __init__(self, input_dim, hidden_size, hidden_mlp_size, word_embeddings_model, pos_embeddings_model):
        """
        Implementation of the dependency parser model that was given in the article.
        Two layer bidirectional lstm for the word and, it's pos embedding.
        Then, every pair of two words and their embeddings will pass to the score object to calculate the score
        of the edge between those words for the chu liu edmonds algo which will give us the m.s.t that will be our
        dependency model for the sentence.
        @param input_dim: The size of the word's and, it's pos embedding.
        @param hidden_size: The hidden size of the lstm.
        @param hidden_mlp_size: The hidden size we will pass tho the score object.
        @param word_embeddings_model: The word embedding model that was built in the preprocess part.
        @param pos_embeddings_model:
        """
        super(DependencyParser, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Pre-trained embedding matrix
        word_weights = torch.FloatTensor(word_embeddings_model.wv.vectors)
        pos_weights = torch.FloatTensor(pos_embeddings_model.wv.vectors)
        self.word_embedding = nn.Embedding.from_pretrained(word_weights, freeze=False).to(self.device)
        self.pos_embedding = nn.Embedding.from_pretrained(pos_weights, freeze=False).to(self.device)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=2, bidirectional=True,
                            batch_first=False)
        # FILL THE SCORING MST HERE
        self.edge_scorer = Score(hidden_size, hidden_mlp_size, self.device)
        self.decoder = decode_mst

    def forward(self, sentence, train=True, is_labeled=True):
        """
        Gets the sentence pass it through the network and move it forward to the score model.
        @param sentence: The sentence we will find it's, dependency parser.
        @param train: For the dropout in the score.
        @param is_labeled: For generating the comp file.
        @return: The dependency parser prediction and the loss.
        """
        words_idx, poss_idx, true_tree_heads = sentence
        word_embeds = self.word_embedding(words_idx.to(self.device))  # [batch_size, seq_length, emb_dim]
        pos_embeds = self.pos_embedding(poss_idx.to(self.device))
        embeds = torch.cat((word_embeds, pos_embeds), 2)
        lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]
        lstm_out = lstm_out[:, 0, :]
        n = lstm_out.shape[0]
        score_matrix, scores_matrix = self.edge_scorer.forward(lstm_out, train)
        prediction, _ = self.decoder(score_matrix, n, has_labels=False)

        if is_labeled:
            loss = calc_loss(true_tree_heads[0].to(self.device), scores_matrix.to(self.device))
        else:
            loss = None
        return prediction, loss
