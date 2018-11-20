#!/usr/bin/python3

import torch
import sys, getopt
import dataset

from models.ResNet import *

def evaluate_model(filepath):
    # load desired model
    print("Loading model at %s" % filepath)
    model = resnet_18()
    model.load_state_dict(torch.load(filepath))
    model.eval()

    # get test data
    batch_size = 100
    _, test_loader = dataset.get_val_test_loaders(batch_size)

    # get test predictions
    outname = 'output.txt'
    f = open(outname, 'w')
    print("Writing output to %s" % outname)

    image_index = 1
    image_len = 8
    base_image_name = "00000000"
    for data in test_loader:
        images, _ = data
        prediction = model(images)
        _, best_five = torch.topk(prediction, 5)

        # create image filename
        imname = base_image_name + str(image_index)
        if len(imname) > image_len:
            imname = imname[len(imname)-image_len:]
        imname += ".jpg"
        imname = "test/" + imname
        image_index += 1
        
        # format predictions into filename 1 2 3 4 5 output file
        for i in range(len(prediction)):
            line = imname + " "
            for j in range(len(prediction[i])):
                line += str(prediction[i][j]) + " "
            f.write(line)
        f.close()

if __name__=='__main__':
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    if len(sys.argv) < 2:
        print("Expected: evaluate_model.py <model_filepath>")
        sys.exit(2)
    evaluate_model(sys.argv[1])

