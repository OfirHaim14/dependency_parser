from matplotlib import pyplot as plt
from models import DependencyParser
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
# CUDA_LAUNCH_BLOCKING=1

# The sizes we chose for the embeddings and networks layers.
WORD_EMBEDDING_SIZE = 200
POS_EMBEDDING_SIZE = 40
HIDDEN_DIM = 300
HIDDEN_MLP_DIM = 150  # was 125 for 89%
EPOCHS = 30


def train_process(train_sentences, test_sentences, word_model, pos_model, weights_path):
    """
    Creates the model, the data loader object for the train loop so, it will only do the train part.
    @param train_sentences: The sentences from the train.
    @param test_sentences: The sentences from the test.
    @param word_model: The embedding  for the sentences model we create in the preprocess part.
    @param pos_model:The embedding  for the pos model we create in the preprocess part.
    @param weights_path: Where will the weights will be saved, will pass to the train loop function.
    @return:
    """
    model = DependencyParser(WORD_EMBEDDING_SIZE+POS_EMBEDDING_SIZE, HIDDEN_DIM, HIDDEN_MLP_DIM, word_model, pos_model)
    trainloader = DataLoader(dataset=train_sentences, shuffle=True)
    testloader = DataLoader(dataset=test_sentences, shuffle=False)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        model.cuda()
    train_and_inference(trainloader, model, device, testloader, weights_path)


def train_and_inference(train_dataloader, model, device, test_dataloader, weights_path):
    """
    This function trains the bidirectional lstm model on the train sentences and after every epoch infer our model
    on the test part because we want to save the model that gave the best score on the test.
    @param train_dataloader: The data loader object of the train sentences, will be trained on.
    @param model: The dependency parser model.
    @param device: on what the code will run.
    @param test_dataloader: The data loader object of the test sentences, will be inferred.
    @param weights_path: Where will the weights of the best model will be saved.
    """
    # We will be using a simple SGD optimizer to minimize the loss function
    optimizer = optim.Adamax(model.parameters(), lr=0.01)
    acumulate_grad_steps = 15  # This is the actual batch_size, while we officially use batch_size=1
    print("Training Started")
    # initialize the loss and UAS lists and the time variables.
    best_epoch = 1
    total_training_time = 0
    total_evaluation_time = 0
    UAS_list = []
    loss_list = []
    test_UAS_list = []
    test_loss_list = []
    for epoch in range(EPOCHS):  # the train loop
        training_time = time.time()  # so we can know how long was the train
        correct_heads = 0
        heads_number = 0
        i = 0
        for batch_idx, input_data in enumerate(train_dataloader):  # run on the test sentences with their number
            i += 1
            words, poss, true_tree = input_data
            predicted_tree, loss = model.forward(input_data)  # pass the input through the lstm and MLP models
            # optimization
            loss = loss / acumulate_grad_steps
            loss.backward()
            if i % acumulate_grad_steps == 0:
                optimizer.step()
                model.zero_grad()
            true_idx_tensor = torch.eq(true_tree[0][1:].to(device),
                                       torch.tensor(predicted_tree[1:], requires_grad=False).to(device))
            correct_heads += true_idx_tensor.sum().item()
            heads_number += true_tree.shape[1] - 1
        # evaluate our UAS and saving it
        total_training_time += time.time() - training_time
        uas = correct_heads / heads_number
        loss_list.append(float(loss.item()))
        UAS_list.append(float(uas))
        # Evaluation
        evaluation_time = time.time()
        test_loss, test_uas = evaluate(model, test_dataloader)
        total_evaluation_time += time.time() - evaluation_time
        test_loss_list.append(float(test_loss))
        test_UAS_list.append(float(test_uas))
        print("Epoch {} Completed!".format(epoch + 1))
        print("TRAIN: Loss = {}, UAS = {}".format(loss_list[-1], UAS_list[-1]))
        print("TEST: Loss = {}, UAS = {}".format(test_loss_list[-1], test_UAS_list[-1]))
        if test_UAS_list[-1] == max(test_UAS_list):
            best_epoch = epoch + 1
            torch.save(model.state_dict(), weights_path)
    plot_eval(UAS_list, loss_list, test_UAS_list, test_loss_list)
    print("\nTotal training time is: " + str(total_training_time))
    print("Total evaluation time is: " + str(total_evaluation_time))
    print("Epoch with best accuracy is: " + str(best_epoch))
    print("Best test accuracy is: " + str(max(test_UAS_list)))


def evaluate(model, test_loader):
    """
    Infer on the test.
    @param model: The dependency parser model.
    @param test_loader: The data loader object of the test sentences, will be inferred.
    @return: The total loss and the UAS
    """
    total_loss = 0
    correct_heads = 0
    heads_number = 0
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        model.cuda()
    with torch.no_grad():
        for batch_idx, input_data in enumerate(test_loader):
            words_idx, pos_idx, true_tree = input_data
            predicted_tree, loss = model.forward(input_data, train=False)
            total_loss += loss
            true_idx_tensor = torch.eq(true_tree[0][1:].to(device),
                                       torch.tensor(predicted_tree[1:], requires_grad=False).to(device))
            correct_heads += true_idx_tensor.sum().item()
            heads_number += true_tree.shape[1] - 1
        uas = correct_heads / heads_number
    return total_loss / len(test_loader), uas


def plot_eval(uas_list, loss_list, test_uas_list, test_loss_list):
    plt.plot(range(1, EPOCHS + 1), uas_list, c="red")
    plt.title("UAS Accuracy on Train")
    plt.xlabel("Epochs")
    plt.ylabel("UAS")
    plt.savefig("train_uas.png")
    plt.show()
    plt.clf()

    plt.plot(range(1, EPOCHS + 1), loss_list, c="blue")
    plt.title("Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("train_loss.png")
    plt.show()
    plt.clf()

    plt.plot(range(1, EPOCHS + 1), test_uas_list, c="red")
    plt.title("UAS Accuracy on Test")
    plt.xlabel("Epochs")
    plt.ylabel("UAS")
    plt.savefig("test_uas.png")
    plt.show()
    plt.clf()

    plt.plot(range(1, EPOCHS + 1), test_loss_list, c="blue")
    plt.title("Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("test_loss.png")
    plt.show()
    plt.clf()


def evaluate_comp(predicted_text_path, comp_sentences_in_format, weights_path, word_model, pos_model, comp_sentences):
    """
    Infer the comp and write the prediction to a prediction file.
    @param predicted_text_path: Where we will write our prediction.
    @param comp_sentences_in_format: The comp sentences.
    @param weights_path: The path to the weights of the model.
    @param word_model: The word embedding we made.
    @param pos_model: The pos embedding we made.
    @param comp_sentences:  A list of the comp sentences.
    """
    model = DependencyParser(WORD_EMBEDDING_SIZE + POS_EMBEDDING_SIZE, HIDDEN_DIM, HIDDEN_MLP_DIM, word_model,
                             pos_model)
    comploader = DataLoader(dataset=comp_sentences_in_format, shuffle=False)
    text = ""
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    with torch.no_grad():
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        for batch_idx, input_data in enumerate(comploader):
            predicted_tree, _ = model(input_data, False, False)
            predicted_tree = predicted_tree[1:]
            sentence = comp_sentences[batch_idx]
            sentence = sentence[1:]
            for i in range(len(sentence)):
                line = str(i + 1) + '\t' + sentence[i][0] + '\t' + '_' + '\t' + sentence[i][
                    1] + '\t' + '_' + '\t' + '_' + '\t' + str(predicted_tree[i]) + '\t' + '_' + '\t' + '_' + '\t' + '_'
                line = line + '\n'
                text = text + line
            text = text + '\n'
        f = open(predicted_text_path, "w")
        f.write(text)
        f.close()
