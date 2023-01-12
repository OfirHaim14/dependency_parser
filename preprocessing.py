import pandas as pd
import torch
import csv
from gensim.models import Word2Vec

# Paths
TRAIN_PATH = './data/train.labeled'
TEST_PATH = './data/test.labeled'
COMP_PATH = './data/comp.unlabeled'

# Different token
ROOT = '<root>'
UNKNOWN = '<unknown>'
PAD = '<pad>'
ROOT_HEAD = -1

# Word2Vec model path name
WORD_EMBEDDING_PATH = 'word2vec.model'
POS_EMBEDDING_PATH = 'pos2vec.model'
WORD_EMBEDDING_SIZE = 200
POS_EMBEDDING_SIZE = 40  # was 25


def merge_files(path1, path2, output_path):
    """
    To train on the train and the test too we merge the two files to one.
    @param path1: The first file
    @param path2: The second
    @param output_path: Will be the combined of the two files.
    """
    # Reading data from file1
    with open(path1) as fp:
        data = fp.read()
    # Reading data from file2
    with open(path2) as fp:
        data2 = fp.read()
    # Merging 2 files
    # To add the data of file2
    # from next line
    data += "\n"
    data += data2
    with open(output_path, 'w') as fp:
        fp.write(data)
    print("finished")


def get_sentences_vocab(path, tagged=True):
    """
    This function returns our sentences as a list of lists, our heads, the vocabulary, and the words and POS separately
    Every word in a sentence represented as a tuple of the word and its POS
    @param path: The file to make vocab from
    @param tagged: to sperate the train, test from comp file.
    """
    print("Started extracting sentences from path: " + path)
    vocab = [ROOT, UNKNOWN]
    pos_vocab = [ROOT, UNKNOWN]
    # Generate a list of sentences and their heads
    sentences = []
    sentences_heads = []
    sentences_words_only = []
    sentences_POS_only = []
    # Temporary lists for each sentence
    current_sentence = [(ROOT, ROOT)]
    current_heads = [ROOT_HEAD]
    current_sentence_words_only = [ROOT]
    current_sentence_POS_only = [ROOT]

    empty_idx = []
    # Finding all the blank lines
    with open(path) as file:
        for (i, line) in enumerate(file):
            if line == "\n":
                empty_idx += [i]

    data = pd.read_csv(path, sep='\t', header=None, quoting=csv.QUOTE_NONE, skip_blank_lines=False)
    data.columns = ['count', 'token', '_', 'POS', '_', '_', 'thead', 'label', '_', '_']

    for i in range(data.shape[0]):
        if i not in empty_idx:
            current_sentence += [(data.loc[i].token, data.loc[i].POS)]
            current_sentence_words_only += [data.loc[i].token]
            current_sentence_POS_only += [data.loc[i].POS]
            if tagged:
                current_heads += [int(data.loc[i].thead)]
            if data.loc[i].token not in vocab:
                vocab.append(data.loc[i].token)
            if data.loc[i].POS not in pos_vocab:
                pos_vocab.append(data.loc[i].POS)
        else:
            sentences.append(current_sentence)
            sentences_words_only.append(current_sentence_words_only)
            sentences_POS_only.append(current_sentence_POS_only)
            if tagged:
                sentences_heads.append(current_heads)
            current_sentence = [(ROOT, ROOT)]
            current_sentence_words_only = [ROOT]
            current_sentence_POS_only = [ROOT]
            current_heads = [ROOT_HEAD]
    if len(current_sentence) != 0:
        sentences.append(current_sentence)
        sentences_words_only.append(current_sentence_words_only)
        sentences_POS_only.append(current_sentence_POS_only)
        if tagged:
            sentences_heads.append(current_heads)
    print("Finished extracting sentences from path")
    return sentences, sentences_heads, vocab, pos_vocab, sentences_words_only, sentences_POS_only


def get_embeddings_model(sentences, path, size):
    sentences = [[w.lower() for w in sen] for sen in sentences]
    sentences += [[UNKNOWN]]
    model = Word2Vec(sentences=sentences, vector_size=size, window=5, min_count=1, workers=4)
    model.save(path)


def get_sentences_in_format(word_sentences, pos_sentences, head_sentences, tagged, word_model, pos_model):
    sentences_in_format = []
    if tagged:
        for i, sen in enumerate(word_sentences):
            words = word_sentences[i]
            poss = pos_sentences[i]
            heads = torch.tensor(head_sentences[i])
            if torch.equal(heads, torch.tensor([-1])):
                print('heads list is [-1] at i=' + str(i))
            words = remove_oov(words, word_model)
            poss = remove_oov(poss, pos_model)
            words_idx = torch.tensor([word2idx(w.lower(), word_model) for w in words])
            poss_idx = torch.tensor([word2idx(p.lower(), pos_model) for p in poss])
            sentences_in_format.append((words_idx, poss_idx, heads))
    else:
        for i, sen in enumerate(word_sentences):
            words = word_sentences[i]
            poss = pos_sentences[i]
            words = remove_oov(words, word_model)
            poss = remove_oov(poss, pos_model)
            words_idx = torch.tensor([word2idx(w.lower(), word_model) for w in words])
            poss_idx = torch.tensor([word2idx(p.lower(), pos_model) for p in poss])
            sentences_in_format.append((words_idx, poss_idx, []))
    return sentences_in_format


def word2idx(word, word_model):
    return torch.tensor(word_model.wv.get_index(word))


def remove_oov(words_list, model):
    for i, word in enumerate(words_list):
        if not model.wv.has_index_for(word.lower()):
            words_list[i] = UNKNOWN
    return words_list