import train, test, utils

n_epochs = 30

if __name__ == "__main__":
    # let's train our model for n_epochs
    best_accuracy, test_loader, train_losses, val_accuracies = train.train(n_epochs)
    print('Finished Training')
    print('The best validation accuracy was %d %%' % best_accuracy)

    # let's test our model on our real dataset
    ground_truth,predictions, accuracy = test.test("Checkpoints/BestModel.pth", test_loader)

    # plot our training loss and validation accuracy
    utils.plot_info(train_losses, val_accuracies)

    # plot confusion matrix of the test set
    utils.plot_confusion_matrix(ground_truth, predictions)
