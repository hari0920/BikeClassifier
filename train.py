""" 
Code to Train a Tensorflow Classifier for mountain bikes and road bikes
Author: Hariharan Ramshankar

"""
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
import argparse
import os
import glob
"""Create a simple parser to handle our custom(optional) inputs, if given"""
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", help="path to training images directory")
parser.add_argument("--test_dir", help="path to testing images directory")
args=parser.parse_args()
if args.train_dir:
    print(args.train_dir)
    train_dir=args.train_dir
if args.test_dir:
    print(args.test_dir)
    test_dir = args.test_dir
else:
    print("Assuming train and test images in default directories.")
    train_dir = os.getcwd()+"/training_images/*/*.jpg"
    test_dir = os.getcwd()+"/testing_images/*.jpg"

#print(os.getcwd())

#Using TF queueing for input
train_filenames = tf.train.string_input_producer(glob.glob(train_dir))
test_filenames = tf.train.string_input_producer(glob.glob(test_dir))

#Creating a Reader for use with Queue
image_reader=tf.WholeFileReader()

#Now read the actual file, ignoring filename
_,training_image_undecoded = image_reader.read(train_filenames)

training_image = tf.image.decode_jpeg(training_image_undecoded)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #initialize all variables
    coordinator=tf.train.Coordinator() #coordinator fot all threads
    threads= tf.train.start_queue_runners(coord=coordinator) #run the threads for input processing
    image_tensor=sess.run(training_image)
    print(image_tensor)
    #stop coordinator and threads
    coordinator.request_stop()
    coordinator.join(threads) #blocks till all terminate

