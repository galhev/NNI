import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(in_features = input_size, out_features = hidden_size, bias=True)
        self.fc2 = nn.Linear(in_features = hidden_size, out_features = num_classes, bias=True)

    def forward(self, text, text_lengths):
        text = text.float() # dense layer deals just with float type data
        x = self.fc1(text)
        preds = self.fc2(x)
        return preds
