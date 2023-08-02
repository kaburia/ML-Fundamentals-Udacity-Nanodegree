import torch
import torch.nn as nn


# define the CNN architecture
# class MyModel(nn.Module):
#     def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

#         super().__init__()
        

#         # YOUR CODE HERE
#         # Define a CNN architecture. Remember to use the variable num_classes
#         # to size appropriately the output of your classifier, and if you use
#         # the Dropout layer, use the variable "dropout" to indicate how much
#         # to use (like nn.Dropout(p=dropout))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # YOUR CODE HERE: process the input tensor through the
#         # feature extractor, the pooling and the final linear
#         # layers (if appropriate for the architecture chosen)
#         return x
    
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()

        # Define a CNN architecture with two convolutional layers
        self.features = nn.Sequential(
            # layer 1
            nn.Conv2d(3, 32, kernel_size=3,  padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # layer 2
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # calculate the number of output features from the CNN for the fully connected layer
        # input image = 224x224
        # 2 MaxPooling layers with 2x2 window and stride 2
        # after the first layer = 112x112
        # after the second layer = 56x56
        
        self.num_features = 64 * 56 * 56

        # Define a fully connected (linear) layer as the classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_features, 256),  # Fully connected layer with 256 output units
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),  # Fully connected layer with num_classes output units
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process the input tensor through the feature extractor
        x = self.features(x)

        # Flatten the tensor before passing it to the classifier
        x = torch.flatten(x, 1)

        # Process the flattened tensor through the classifier
        x = self.classifier(x)
        
        return x

######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
