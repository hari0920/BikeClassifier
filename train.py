""" 
Code to Train a Tensorflow Classifier for mountain bikes and road bikes
Author: Hariharan Ramshankar

"""
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
import argparse

"""Create a simple parser to handle our custom(optional) inputs, if given"""
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", help="path to training images directory")
parser.add_argument("--test_dir", help="path to testing images directory")
args=parser.parse_args()
if args.train_dir:
    print(args.train_dir)
if args.test_dir:
    print(args.test_dir)
else:
    print("Assuming train and test images in default directories.")

