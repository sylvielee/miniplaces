#!/usr/bin/python3

import torch
import sys, getopt
import os
import dataset
import matplotlib.pyplot as plt

from models.ResNet import *

def evaluate_model(filepath):
    # load desired model
    print("Loading model at %s" % filepath)
    model = resnet_18(0)
    model.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
    model.eval()

    # get output file dir
    output_dir = os.path.dirname(filepath)

    # get test data
    batch_size = 100
    vl, test_loader = dataset.get_val_test_loaders(batch_size)
    classes = vl.dataset.classes

    # get test predictions
    outname = output_dir + '/output.txt'
    f = open(outname, 'w')
    print("Writing output to %s" % outname)

    image_index = 1
    image_len = 8
    base_image_name = "00000000"
    print("Processing %d batches" % len(test_loader))
    print(test_loader.dataset.classes)

    for data in test_loader:
        images, _ = data[:10]
        prediction = model(images)
        _, best_five = torch.topk(prediction, 5)

        # format predictions into filename 1 2 3 4 5 output file
        for i in range(len(prediction)):
            # create image filename
            imname = base_image_name + str(image_index)
            if len(imname) > image_len:
                imname = imname[len(imname)-image_len:]
            imname += ".jpg"
            imname = "test/" + imname
            image_index += 1

            line = imname + " "
            for j in range(len(best_five[i])):
                actual_class = classes[best_five[i][j].item()]
                line += str(actual_class) + " "
            line += "\n"
            f.write(line)
        print("Up to image %d" % image_index)

    f.close()

if __name__=='__main__':
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    if len(sys.argv) < 2:
        print("Expected: evaluate_model.py <model_filepath>")
        sys.exit(2)
    evaluate_model(sys.argv[1])




