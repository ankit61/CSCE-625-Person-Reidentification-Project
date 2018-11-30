#!/usr/bin/env python3
# coding: utf-8

import sys, os
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix

# Tensorboard 
from tensorboardX import SummaryWriter

class ScaleDownOrPad(object):
    
    
    def __init__(self, output_size, interpolation=Image.BILINEAR, fill=0):
        self.interpolation = interpolation
        self.fill = fill
        self.output_size = output_size
        if output_size[0] > output_size[1]:
            self.greater_side_size = output_size[0]
        else:
            self.greater_side_size = output_size[1]
        
    def __call__(self, input):
        w, h = input.size
        if w > self.output_size[0] or h > self.output_size[1]:
            if w > h:
                ow = self.greater_side_size
                oh = int( self.greater_side_size * h / w )
                input = input.resize((ow, oh), self.interpolation)

            else:
                oh = self.greater_side_size
                ow = int(self.greater_side_size * w / h)
                input =  input.resize((ow, oh), self.interpolation)
        
        input_position = (np.asarray(self.output_size) // 2) - (np.asarray(input.size) // 2)

        output = Image.new(mode=input.mode,
                           size=self.output_size,
                           color=self.fill)
        
        output.paste(input, box=tuple(input_position))
        
        return output

func = ScaleDownOrPad((244,244)) 

img = Image.open("example.jpg").convert('RGB')

img = func(img)

img.save("test.png")






