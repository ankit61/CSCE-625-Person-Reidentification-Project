#!/usr/bin/env python3
from network import start_training #, process_images

def train():
    #process_images("/datasets/DukeMTMC-reID/bounding_box_test/", "/datasets/DukeSegmented/test/")
    start_training(
        "/datasets/LIP/TrainVal_images/train_images/",
        "/datasets/LIP/EDITED_TrainVal_parsing_annotations/train_segmentations/",        
        "/datasets/LIP/TrainVal_images/val_images/",
        "/datasets/LIP/EDITED_TrainVal_parsing_annotations/val_segmentations/",
        epochs=100
    )

if __name__=="__main__":
    train()

