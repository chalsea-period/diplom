import custom_datasets as cd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import neural_network_class as nnc
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from time import time

torch.cuda.empty_cache()

training_data=cd.dataset_1(target="train")
train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
# test_data=cd.dataset_1(target="test")
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
# for train_features, train_labels,train_restinfo in iter(train_dataloader):
#     print(f"Feature batch shape: {train_features.size()}")
#     print(f"Labels batch shape: {train_labels.size()}")
#     img = train_features[0].squeeze()
#     img = img.permute(1, 2, 0)
#     label = train_labels[0]
#     plt.imshow(img)
#     plt.show()
#     print(f"Label: {label}")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
nnc_model=nnc.NeuralNetwork().to(device)
flatten = nn.Flatten()
layer_1 = nn.Linear(in_features=640 * 640, out_features=20)

softmax = nn.Softmax(dim=1)

epochs=20
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(nnc_model.parameters(), lr=1e-3)

t=time()
acc_list=[]
for epoch in range(epochs):
    print("#"*int((epoch)/(epochs)*100*2+3))
    print(f"starting poch {epoch + 1}/{epochs}  {((epoch + 1)/(epochs))*100}%")
    c = 0
    f=0
    for img, label, x_cent, y_cent, w, h in train_dataloader:
        img=img.to(device)
        label=label.to(device)
        with torch.autocast(device_type="cuda"):
            logits=nnc_model(img)

        probs = softmax(logits)
        predicted_classes = logits.argmax(dim=1)
        confidences = probs.max(dim=1).values
        print(logits.shape,probs.shape,'\n',
              confidences,'\n',
              predicted_classes,'\n',
              label)

        for j in range(len(predicted_classes)):
            f += 1
            if predicted_classes[j]==label[j]:
                c+=1

        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc_list.append(str(c / f * 100))
    print(f"Loss: {loss.item():.4f}")
    print(f"accuracy on epoch: {c / f * 100}%")

print(f"total elapsed time {int((time()-t)//60)}:{(time()-t)%60}")
print(f"accuracy stat {" ".join(acc_list)}")