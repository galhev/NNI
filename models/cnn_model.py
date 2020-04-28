import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, n_filters, filter_sizes, pool_size, hidden_size, num_classes,
                 dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1,
                                              out_channels=n_filters,
                                              kernel_size=(fs, embed_size))
                                    for fs in filter_sizes])
        self.max_pool1 = nn.MaxPool1d(pool_size) # pool size = 2
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(95*n_filters, hidden_size, bias=True)  # dense  # TODO: fixed_length to dynamic batch - more efficient in calculations
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=True)  # dense layer

    def forward(self, text, text_lengths):
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1) # Conv1d gets 4 dimension
        convolution = [conv(embedded) for conv in self.convs]

        max1 = self.max_pool1(convolution[0].squeeze()) # pooling size = 2
        max2 = self.max_pool1(convolution[1].squeeze())

        cat = torch.cat((max1, max2), dim=2)
        x = cat.view(cat.shape[0], -1)  # x = Flatten()(x)
        x = self.fc1(self.relu(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


