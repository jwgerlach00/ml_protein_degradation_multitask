import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class RegMultitask(nn.Module):
    @classmethod
    def plot_graph(cls, bit_vector_length:int, num_targets:int) -> None:
        X = torch.zeros((1, bit_vector_length))
        model = cls(bit_vector_length, num_targets, torch.device('cpu'))
        writer = SummaryWriter("torchlogs/")
        writer.add_graph(model, X)
        writer.close()

    def __init__(self, bit_vector_length:int, num_targets:int, device:torch.device) -> None:
        super().__init__()
        self.__device = device
        self.__num_targets = num_targets

        self.linear1 = nn.Linear(bit_vector_length, 45)
        self.linear2 = nn.Linear(45, 12)

        # By setting attributes we can avoid having to pass device to each layer in the list, it will just reference \
            # the model device
        [setattr(self, f'out_linear_layer_{i}', nn.Linear(12, 1)) for i in range(num_targets)]

        self.relu = nn.ReLU() # activation
        self.dropout = nn.Dropout(0.2)

    def forward(self, X) -> torch.Tensor:
        X = self.dropout(self.relu(self.linear1(X)))
        X = self.dropout(self.relu(self.linear2(X)))
        X = [getattr(self, f'out_linear_layer_{i}')(X) for i in range(self.__num_targets)] # tensor list
        X = torch.cat(X, axis=1)
        X.to(self.__device)
        return X
