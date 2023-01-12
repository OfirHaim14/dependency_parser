from train_and_eval import train_process, evaluate_comp
from preprocessing import get_sentences_vocab, get_sentences_in_format, get_embeddings_model, merge_files
from gensim.models import Word2Vec

# Paths
TRAIN_PATH = './data/train.labeled'
TEST_PATH = './data/test.labeled'
COMP_PATH = './data/comp.unlabeled'
TEST_AND_TRAIN_PATH = 'test_and_train.labeled'
PRED_COMP_PATH = "comp_213496110_214169377.labeled"
weights_path = "model_weights"
WORD_EMBEDDING_PATH = 'word2vec.model'
POS_EMBEDDING_PATH = 'pos2vec.model'

# The size of the embeddings
WORD_EMBEDDING_SIZE = 200
POS_EMBEDDING_SIZE = 40


# for better results we decided to train the model on both the test and the train so we merged them to one file.
# we got the vocab of the train and test.
merge_files(TRAIN_PATH, TEST_PATH, TEST_AND_TRAIN_PATH)
_, train_test_heads, train_test_word_vocab, train_test_pos_vocab, train_test_words, train_test_POS = \
    get_sentences_vocab(TEST_AND_TRAIN_PATH, tagged=True)
_, test_heads, test_word_vocab, test_pos_vocab, test_words, test_POS = get_sentences_vocab(TEST_PATH, tagged=True)

# we made embedding for the words and it's pos
get_embeddings_model(train_test_words, WORD_EMBEDDING_PATH, WORD_EMBEDDING_SIZE)
get_embeddings_model(train_test_POS, POS_EMBEDDING_PATH, POS_EMBEDDING_SIZE)
word_model = Word2Vec.load(WORD_EMBEDDING_PATH)
pos_model = Word2Vec.load(POS_EMBEDDING_PATH)

# we get the words
train_test_in_format = get_sentences_in_format(train_test_words, train_test_POS, train_test_heads, True, word_model, pos_model)
test_in_format = get_sentences_in_format(test_words, test_POS, test_heads, True, word_model, pos_model)
train_process(train_test_in_format, test_in_format, word_model, pos_model, weights_path)


comp_sentences, _, comp_word_vocab, comp_pos_vocab, comp_words, comp_POS = get_sentences_vocab(COMP_PATH, tagged=False)
comp_in_format = get_sentences_in_format(comp_words, comp_POS, [], False, word_model, pos_model)
evaluate_comp(PRED_COMP_PATH, comp_in_format, weights_path, word_model, pos_model, comp_sentences)
