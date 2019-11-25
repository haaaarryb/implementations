import dataset
import torch
from torchvision import transforms
from torch.utils.data import random_split, DataLoader


mytransforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dset = dataset.ImgDataset(transform=mytransforms)
print(dset[0])

train_data, val_data, test_data = random_split(dset, [7, 3, 1])

train_loader = DataLoader(train_data,
                          shuffle=True,
                          batch_size=4)

class Autoencoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 5),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, 5),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 64, 5),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 3, 5),
            torch.nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        x_tilde = self.decoder(z)
        return x_tilde


mymodel = Autoencoder()

criterion = torch.nn.MSELoss()
optimiser = torch.optim.Adam(params=mymodel.parameters(), lr=0.01)


def train(model, epochs=10):
    for epoch in range(epochs):
        for x in train_loader:
            x_til = model(x)
            loss = criterion(x_til, x)
            print(loss)
            lk
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            print('Loss:', loss.item())

train(mymodel)


encoding = mymodel.encode(newinput)



