import gc
import os
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

def run(model_name, cuda_num, dropout_rate, lr, momentum=0):
    # Parameters
    num_epochs = 10
    output_period = 10
    batch_size = 100

    # setup the device for running
    device = torch.device("cuda:%s" % cuda_num if torch.cuda.is_available() else "cpu")
    model = resnet_18(dropout_rate)
    model = model.to(device)

    train_loader, val_loader = dataset.get_data_loaders(batch_size)
    num_train_batches = len(train_loader)

    criterion = nn.CrossEntropyLoss().to(device)
    
    # TODO for part 1: try lr = 1e-2 and 1e-1
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # TODO for part 2
    # lr_decay = lr/num_epochs
    # optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay) 
    # optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08) # try diff betas maybe (0.97, 0.98)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum) # try momentums from 0.5-0.8 in .1 steps

    epoch = 1

    # save errors from each epoch
    train_class_errors = []
    train_top5_errors = []
    val_class_errors = []
    val_top5_errors = []

    # will save errors from each epoch
    training_vals_folder = 'training_values/'
    f = open(training_vals_folder + model_name + '_training_values.txt', 'w')

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
        try:
            os.stat("models/" + model_name)
        except:
            os.mkdir("models/" + model_name)       
        torch.save(model.state_dict(), "models/%s/model.%d" % (model_name, epoch))

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
        f.write("\nEpoch %d\n\tTraining classification error %0.3f\n\tTraining top5 error %0.3f" % (epoch, tn_class_err, tn_top5_err))

        train_class_errors.append(tn_class_err)
        train_top5_errors.append(tn_top5_err)

        print("Training Dataset of size %d \n\tClassification Err: %0.3f\n\tTop-5 Err: %0.3f" % (tn_total, tn_class_err, tn_top5_err))

        # validation dataset classification error
        model.eval()
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
        f.write("\n\tValidation classification error %0.3f\n\tValidation top5 error %0.3f" % (epoch, val_class_err, val_top5_err))

        val_class_errors.append(val_class_err)
        val_top5_errors.append(val_top5_err)

        print("Validation Dataset of size %d \n\tClassification Err: %0.3f\n\tTop-5 Err: %0.3f" % (val_total, val_class_err, val_top5_err))
        gc.collect()
        epoch += 1

    f.close()

    # visualize errors
    image_folder = 'imgs/'
    xaxis = [i for i in range(num_epochs)]
    plt.plot(xaxis, train_class_errors, xaxis, val_class_errors)
    plt.savefig(image_folder + model_name + '_classification_comparison.png')
    plt.clf()

    plt.plot(xaxis, train_top5_errors, xaxis, val_top5_errors)
    plt.savefig(image_folder + model_name + '_top5_comparison.png')

if __name__=='__main__':
    if len(sys.argv) < 5:
        print("Expected: train.py <model_name> <cuda_num> <dropout_rate> <learning_rate>, momentum is optional")
        sys.exit(2)

    print('Starting training')
    if len(sys.argv) == 5: # no momentum
        run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    print('Training terminated')