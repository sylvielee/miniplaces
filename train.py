import gc
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark=True

import dataset
from models.AlexNet import *
from models.ResNet import *

perc = 0
def run():
    # Parameters
    num_epochs = 10
    output_period = 10
    batch_size = 100

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_18()
    # model.change_p(perc)
    model = model.to(device)

    train_loader, val_loader = dataset.get_data_loaders(batch_size)
    num_train_batches = len(train_loader)

    criterion = nn.CrossEntropyLoss().to(device)
    # TODO: optimizer is currently unoptimized
    # there's a lot of room for improvement/different optimizers
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    epoch = 1

    # save errors from each epoch
    train_class_errors = []
    train_top5_errors = []
    val_class_errors = []
    val_top5_errors = []

    # will save errors from each epoch
    f = open('training_values.txt', 'w')

    while epoch <= num_epochs:
        running_loss = 0.0
        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
        model.train()

        for batch_num, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f' % (
                    epoch, batch_num*1.0/num_train_batches,
                    running_loss/output_period
                    ))
                running_loss = 0.0
                gc.collect()

        gc.collect()
        # save after every epoch
        torch.save(model.state_dict(), "models/model.%d" % epoch)

        # TODO: Calculate classification error and Top-5 Error
        # on training and validation datasets here

        tn_class_err, val_class_err = 0, 0
        tn_top5_err, val_top5_err = 0, 0
        tn_total, val_total  = 0, 0
        class_correct, fiveclass_correct = 0, 0

        # training dataset classification error
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            prediction = model(images)
            # prediction = prediction.to('cpu')

            _, top_choices = torch.max(prediction, dim=1)
            _, top5_choices = torch.topk(prediction, 5)

            tn_total += labels.size(0)

            class_correct += (top_choices == labels).sum().item()
            for i in range(labels.size(0)):
                if labels[i] in top5_choices[i]:
                    fiveclass_correct += 1

        tn_class_err = 1- class_correct/tn_total
        tn_top5_err = 1- fiveclass_correct/tn_total

        # write to file
        f.write("\nEpoch %d training classification error %0.3f\n\ntraining top5 error %0.3f" % (epoch, tn_class_err, tn_top5_err))

        train_class_errors.append(tn_class_err)
        train_top5_errors.append(tn_top5_err)

        print("Training Dataset of size %d \n\tClassification Err: %0.3f\n\tTop-5 Err: %0.3f" % (tn_total, tn_class_err, tn_top5_err))

        # model.change_p(0)
        model.eval()
        # validation dataset classification error
        class_correct, fiveclass_correct = 0, 0
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            prediction = model(images)
            # prediction = prediction.to('cpu')

            _, top_choices = torch.max(prediction, dim=1)
            _, top5_choices = torch.topk(prediction, 5)

            val_total += labels.size(0)

            class_correct += (top_choices == labels).sum().item()
            for i in range(labels.size(0)):
                if labels[i] in top5_choices[i]:
                    fiveclass_correct += 1

        val_class_err = 1 - class_correct/val_total
        val_top5_err = 1 - fiveclass_correct/val_total

        # write to file
        f.write("Epoch %d validation classification error %0.3f\n\validation top5 error %0.3f" % (epoch, val_class_err, val_top5_err))

        val_class_errors.append(val_class_err)
        val_top5_errors.append(val_top5_err)

        print("Validation Dataset of size %d \n\tClassification Err: %0.3f\n\tTop-5 Err: %0.3f" % (val_total, val_class_err, val_top5_err))
        # model.change_p(perc)
        gc.collect()
        epoch += 1

    f.close()

    # visualize errors
    xaxis = [i for i in range(num_epochs)]
    plt.plot(xaxis, train_class_errors, xaxis, val_class_errors)
    plt.savefig('./classification_comparison.png')
    plt.clf()

    plt.plot(xaxis, train_top5_errors, xaxis, val_top5_errors)
    plt.savefig('./top5_comparison.png')

print('Starting training')
run()
print('Training terminated')
