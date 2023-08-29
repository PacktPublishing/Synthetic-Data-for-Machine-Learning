import torch
import Model
import itertools
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


# ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()  # interactive mode
batch_size = 8

# let's do some data augmentation to improve our synthetic dataset
trans_train = transforms.Compose([
  transforms.Resize([500,600]),
  transforms.CenterCrop([300,300]),
  transforms.RandomHorizontalFlip(),
  transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
  transforms.ToTensor(),])

trans_val_test = transforms.Compose([
  transforms.Resize([500, 600]),
  transforms.CenterCrop([300, 300]),
  transforms.ToTensor(),])

cars_train = ImageFolder("/home/kerim/DataSets/CCIH/train", transform=trans_train)

# No data augmentation on the validation and test splits
cars_val = ImageFolder("/home/kerim/DataSets/CCIH/valid", transform=trans_val_test)
cars_test = ImageFolder("/home/kerim/DataSets/CCIH/test", transform=trans_val_test)


train_loader = torch.utils.data.DataLoader(cars_train, batch_size=batch_size, shuffle=True,  num_workers=4)
val_loader = torch.utils.data.DataLoader(cars_val, batch_size=batch_size,  num_workers=4)
test_loader = torch.utils.data.DataLoader(cars_test, batch_size=batch_size,  num_workers=4)

# create our ML model
model = Model.Network()

# define the loss function and the optimizer
loss_fn = nn.BCELoss()
optimizer = Adam(model.parameters())

# define our execution device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The model will be running on", device, "device")


# save the model
def save_model(epoch):
    # path = "BestModel_"+str(epoch)+".pth"
    path = "Checkpoints/BestModel.pth"
    torch.save(model.state_dict(), path)


# test the model with the test dataset and print the accuracy for the test images
def valid_accuracy():
    model.eval()

    total = 0.0
    correct = 0
    with torch.no_grad():
        for (images, labels) in val_loader:
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # run the model on the test set to predict labels
            outputs = model(images)
            predicted = np.where(outputs.cpu() < 0.5, 0, 1)
            predicted = list(itertools.chain(*predicted))
            total += labels.size(0)
            correct += (predicted == labels.cpu().numpy()).sum().item()

    # compute the accuracy over all test images
    accuracy = 100 * correct // total
    return accuracy


# training function.
# we simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    best_accuracy = 0.0
    losses = []
    accuracy_val = []
    # send our model to CPU or GPU
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()

            # predict classes using images from the training set
            outputs = model(images)

            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels.unsqueeze(-1).float())

            # back-propagate the loss
            loss.backward()

            # adjust parameters based on the calculated gradients
            optimizer.step()

            running_loss += loss.item()  # extract the loss value

        # save training losses to plot them at the end of the training
        losses.append(running_loss)

        # print loss at the end of each epoch
        print('-------------------------------------------')
        print('Epoch %d, train loss: %.3f' % (epoch + 1, running_loss))

        # compute and print the average validation accuracy
        accuracy = valid_accuracy()

        # save validation accuracy to plot them at the end of the training
        accuracy_val.append(accuracy)

        print('The validation accuracy on the valid set (synth + real) is %d %%' % accuracy)

        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            print('Best validation accuracy on the valid set (synth + real) is %d %%' % accuracy)
            save_model(epoch)
            best_accuracy = accuracy

    return best_accuracy, test_loader, losses, accuracy_val
