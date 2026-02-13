import torch
import custom_datasets as cd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

training_data=cd.dataset_1(target="train")
test_data=cd.dataset_1(target="test")

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
for train_features, train_labels in iter(train_dataloader):
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    img = img.permute(1, 2, 0)
    label = train_labels[0]
    plt.imshow(img)
    plt.show()
    print(f"Label: {label}")