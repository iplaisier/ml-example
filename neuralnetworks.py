import torch.nn as nn
import torch.nn.functional as F
import torch






class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, num_outputs, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        self.rnn = nn.RNN(embedding_size, hidden_size, batch_first = True)
        self.fc3 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        embed = self.embedding(x) #returns an embedding for each word
        hidden_state = torch.zeros(1, embed.size(0), self.hidden_size)
        h, _ = self.rnn(embed, hidden_state)
        h = h[:, -1, :]
        h = self.fc3(h)
        return h

class SWEM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, num_outputs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        self.fc1 = nn.Linear(embedding_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_outputs)
    def forward(self, x):
        embed = self.embedding(x) #returns an embedding for each word
        embed_mean = torch.mean(embed, dim=1) # we take the mean embedding per review 
        h = self.fc1(embed_mean)
        h = F.relu(h)
        h = self.fc2(h)
        return h

class CNN_digitrec(nn.Module): 
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Conv2d(1, 32, kernel_size=3)
        self.lin = nn.Linear(26*26*32, 10)

    def forward(self, x):
        # conv layer 
        x = self.conv(x)
        x = F.relu(x)

        # lin layer 
        x = x.view(-1, 26*26*32) 
        x = self.lin(x)
        return x   



class CNN_safarirec(nn.Module): 
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Conv2d(3, 32, kernel_size=3)
        self.lin = nn.Linear(64*64*32, 90)

    def forward(self, x):
        # conv layer 
        x = self.conv(x)
        x = F.relu(x)

        # lin layer 
        x = x.view(-1, 64*64*32) 
        x = self.lin(x)
        return x   
