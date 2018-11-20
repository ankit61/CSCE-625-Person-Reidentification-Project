#!/usr/bin/env python3
import network.start_training
def main():
    network.start_training(
        "/datasets/LIP/TrainVal_images/train_images/",
        "/datasets/LIP/EDITED_TrainVal_parsing_annotations/train_segmentations/",        
        "/datasets/LIP/TrainVal_images/val_images/",
        "/datasets/LIP/EDITED_TrainVal_parsing_annotations/val_segmentations/",
        epocs=100
    )

if __name__=="__main__":
    main()
