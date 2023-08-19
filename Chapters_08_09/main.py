import train, test, utils
n_epochs = 20


if __name__ == "__main__":
    # let's train our model for n_epochs
    best_accuracy, test_loader, train_losses, val_accuracies = train.train(n_epochs)
    print('Finished Training')
    print('The best validation accuracy was %d %%' % best_accuracy)
    utils.plot_info(train_losses, val_accuracies)

    # let's test our model on our real dataset
    ground_truth,predictions = test.test("Checkpoints/BestModel.pth",test_loader)

    # plot confusion matrix of the test set
    utils.plot_confusion_matrix(ground_truth, predictions)