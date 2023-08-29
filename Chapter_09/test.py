import torch
import Model
import itertools
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion() # interactive mode
batch_size = 8


# define our execution device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# test the model with the test dataset and print the accuracy for the test images
def test_accuracy(model, test_loader):
    model.eval()
    ground_truth = []
    predictions = []
    total = 0.0
    correct = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # run the model on the test set to predict labels
            outputs = model(images)
            predicted = np.where(outputs.cpu() < 0.5, 0, 1)
            predicted = list(itertools.chain(*predicted))

            total += labels.size(0)
            correct += (predicted == labels.cpu().numpy()).sum().item()
            ground_truth.append(labels.cpu().numpy())
            predictions.append(predicted)

    # compute the accuracy over all test images
    accuracy = 100 * correct // total
    return accuracy,ground_truth, predictions


# show sample images
def image_show(img,ground_truth, predictions):
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title('Ground_truth:'+str(ground_truth)+'\n  Predictions:'+str(predictions))
    plt.axis('off')
    plt.show()


# test the model with a batch of images and show the labels predictions
def test_batch(model,test_loader):
    # get batch of images from the test DataLoader
    images, labels = next(iter(test_loader))


    # Show the real labels on the screen
    ground_truth = ' '.join('%5s' % np.array(labels.data)[j]
                                    for j in range(batch_size))
    print('Ground truth:', ground_truth)

    # model prediction
    outputs = model(Variable(images.to(device)))
    outputs = np.where(outputs.cpu() < 0.5, 0, 1)

    predictions = ' '.join('%5s' % np.array(outputs.data)[j][0]
                                  for j in range(batch_size))

    print('Predicted:', predictions)

    # show all images as one image grid
    image_show(torchvision.utils.make_grid(images),ground_truth, predictions)


def test(path,test_loader):
    print('The model in path %s will be tested', path)

    # let's load the model we just created and test the accuracy per label
    model = Model.Network()
    model.to(device)

    model.load_state_dict(torch.load(path))
    accuracy,ground_truth, predictions = test_accuracy(model,test_loader)
    print('The test accuracy on the test set (real) is %d %%' % accuracy)

    # test on a random batch of images
    test_batch(model,test_loader)

    ground_truth = np.concatenate(ground_truth, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    return ground_truth, predictions, accuracy