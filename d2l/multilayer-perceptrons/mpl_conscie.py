import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256,10)
)

loss = nn.CrossEntropyLoss(reduction="none")

batch_size = 256
lr = 0.1
num_epochs = 10
trainer = torch.optim.SGD(net.parameters(), lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

d2l.predict_ch3(net, test_iter)


