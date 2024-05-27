import torch.nn as nn

class DoubleQNetwork(nn.Module):
    def __init__(self):
        super(DoubleQNetwork, self).__init__()
        self.conv1 = nn.Sequential(nn.Linear(4, 128), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(128, 1))

        self._create_weights()

    def get_name(self):
        return "Qdoub"
    
    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x