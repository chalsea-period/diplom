import custom_datasets as cd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import neural_network_class as nnc
import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from time import time



def main():
    torch.cuda.empty_cache()

    # training_data=cd.dataset_1(target="train")
    # train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
    # test_data=cd.dataset_1(target="test")
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    training_data = cd.intel_dataset(transform=transforms
                                     .Compose([transforms.Resize((225, 225))]))
    train_dataloader = DataLoader(training_data,
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)

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

    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")
    nnc_model=nnc.NeuralNetwork().bfloat16()
    # nnc_model=torch.load('../src/models/basic_model.pth',weights_only=False)
    nnc_model.to(device)
    flatten = nn.Flatten()
    layer_1 = nn.Linear(in_features=640 * 640, out_features=24)

    softmax = nn.Softmax(dim=1)

    epochs=50
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nnc_model.parameters(), lr=1e-3)

    t=time()
    t1=t
    acc_list=[]
    for epoch in range(epochs):
        print("#"*int((epoch)/(epochs)*100*2+3))
        print(f"starting poch {epoch + 1}/{epochs}  {((epoch + 1)/(epochs))*100}%")
        c = 0
        f=0
        for img, label in train_dataloader:
            img=img.bfloat16().to(device)
            label=label.to(device)

            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                logits=nnc_model(img)

            probs = softmax(logits)
            predicted_classes = logits.argmax(dim=1)
            confidences = probs.max(dim=1).values
            # print(logits.shape,probs.shape,'\n',
            #       confidences,'\n',
            #       predicted_classes,'\n',
            #       label)

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
        print(f"time on epoch {int((time()-t1)//60)}:{(time()-t1)%60}")
        t1=time()

    print(f"total elapsed time {int((time()-t)//60)}:{(time()-t)%60}")
    print(f"accuracy stat {" ".join(acc_list)}")

    torch.save(nnc_model, '../src/models/basic_model_w_classes.pth')

if __name__ == "__main__":
    main()