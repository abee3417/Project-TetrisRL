import torch.nn as nn

class SimpleQNetwork(nn.Module):
    def __init__(self):
        super(SimpleQNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        self._create_weights()

    def get_name(self):
        return "Qsimp"
    
    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.fc(x)