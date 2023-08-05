# load the model that got the best validation accuracy
from src.train import one_epoch_test
from src.model import MyModel
import torch
from src.optimization import get_optimizer, get_loss
from src.data import get_data_loaders

num_classes = 50
dropout=0.5
batch_size =25
valid_size = 0.25

data_loaders = get_data_loaders(batch_size=batch_size, valid_size=valid_size)

model = MyModel(num_classes=num_classes, dropout=dropout)

# YOUR CODE HERE: load the weights in 'checkpoints/best_val_loss.pt'
# model = torch.load('checkpoints/best_val_loss.pt')
model.load_state_dict(torch.load("checkpoints/best_val_loss5.pt", map_location='cpu'))
loss = get_loss()
# Run test
one_epoch_test(data_loaders['test'], model, loss)
