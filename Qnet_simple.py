import torch.nn as nn

class SimpleQNetwork(nn.Module):
    def __init__(self):
        super(SimpleQNetwork, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(4, 32), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(32, 1))

        self._create_weights()

    def get_name(self):
        return "Qsimp"
    
    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x