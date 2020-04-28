import torch
import torch.nn as nn
import os
import nni

from util.data_preprocessing import get_files
from train_test import create_iterator, run_train, evaluate
from models.linear_model import Linear
from models.cnn_model import CNN


if __name__ == "__main__":

    params = nni.get_next_parameter()
    # placing the tensors on the GPU if one is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    path = 'C:/Users/Yaron/PycharmProjects/nni_experiment'
    path_data = os.path.join(path, "data")

    # parameters
    model_type = "CNN"
    data_type = "token" # or: "morph"

    char_based = True
    if char_based:
        tokenizer = lambda s: list(s) # char-based
    else:
        tokenizer = lambda s: s.split() # word-based

    # hyper-parameters:

    batch_size = params['batch_size']
    dropout_keep_prob = params['dropout_keep_prob']
    num_epochs = params["num_epochs"]
    learning_rate = params['lr']
    """
    batch_size = 20
    dropout_keep_prob = 0.2
    num_epochs = 1
    learning_rate = 0.2
    """

    embedding_size = 300
    max_document_length = 100  # each sentence has until 100 words
    dev_size = 0.8 # split percentage to train\validation data
    max_size = 5000 # maximum vocabulary size
    seed = 1
    num_classes = 3

    train_data, valid_data, test_data, Text, Label = get_files(path_data, dev_size, max_document_length, seed, data_type, tokenizer)
    # Create a dictionary
    Text.build_vocab(train_data, max_size=max_size)
    Label.build_vocab(train_data)
    vocab_size = len(Text.vocab)

    train_iterator, valid_iterator, test_iterator = create_iterator(train_data, valid_data, test_data, batch_size, device)
    loss_func = nn.CrossEntropyLoss()
    to_train = True

    if (model_type == "Linear"):
        hidden_size = 100
        model = Linear(max_document_length, hidden_size, num_classes) # input size when there is no embedding layer is max_doc_size and with embedding is the vocabulary size

    if (model_type == "CNN"):
        hidden_size = 128
        pool_size = 2
        n_filters = 128
        filter_sizes = [3, 8]
        model = CNN(vocab_size, embedding_size, n_filters, filter_sizes, pool_size, hidden_size, num_classes, dropout_keep_prob)


    # optimization algorithm
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # train and evaluation
    if (to_train):
        # train and evaluation
        run_train(num_epochs, model, train_iterator, valid_iterator, optimizer, loss_func, model_type)

    # load weights
    saved_file_name = "saved_weights_" + model_type + ".pt"
    model.load_state_dict(torch.load(os.path.join(path, saved_file_name)))
    # predict
    test_loss, test_acc = evaluate(model, test_iterator, loss_func)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    nni.report_final_result(test_acc)

