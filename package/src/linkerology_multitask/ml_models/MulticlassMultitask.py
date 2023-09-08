import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class MulticlassMultitask(nn.Module):
    @classmethod
    def plot_graph(cls, bit_vector_length:int, num_targets:int, num_classes:int) -> None:
        X = torch.zeros((1, bit_vector_length))
        model = cls(bit_vector_length, num_targets, num_classes, torch.device('cpu'))
        writer = SummaryWriter("torchlogs/")
        writer.add_graph(model, X)
        writer.close()

    def __init__(self, bit_vector_length:int, num_targets:int, num_classes:int, device:torch.device) -> None:
        super().__init__()
        self.__device = device
        self.__num_targets = num_targets

        self.linear1 = nn.Linear(bit_vector_length, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)

        # By setting attributes we can avoid having to pass device to each layer in the list, it will just reference \
            # the model device
        [setattr(self, f'out_linear_layer_{i}', nn.Linear(128, num_classes)) for i in range(num_targets)]
        [setattr(self, f'softmax_{i}', nn.Softmax(dim=1)) for i in range(num_targets)]

        self.relu = nn.ReLU() # activation
    
    def forward(self, X) -> torch.Tensor:
        X = self.relu(self.linear1(X))
        X = self.relu(self.linear2(X))
        X = self.relu(self.linear3(X))
        X = [getattr(self, f'softmax_{i}')(getattr(self, f'out_linear_layer_{i}')(X)) for i in
             range(self.__num_targets)] # tensor list
        X = torch.cat(X, axis=1)
        X.to(self.__device)
        return X
