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
import cv2
#from tensorflow.python.framework import ops
#from tensorflow.python.framework import dtypes


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
    if os.name == 'nt':
      print("Windows OS")
      train_dir = os.getcwd()+"\\training_images"
      test_dir = os.getcwd()+"\\testing_images"
    else:
      print("Linux System")
      train_dir = os.getcwd()+"/training_images"
      test_dir = os.getcwd()+"/testing_images"
      
#hyperparameters
learning_rate=0.001
epochs=400
batch_size=20
patch_size=64
num_classes=2
num_parallel_processors=4

#DATA Loading
#obtain the labels
train_categories=os.listdir(train_dir) # we get mountain_bike and road_bike
training_filenames =[]

image_format=".jpg"
category=0
#obtain list of all filepaths
if os.name == 'nt':
    for i in train_categories:
        training_filenames.append(
            glob.glob(train_dir+"\\"+i+"\\*"+image_format))

    test_filenames = glob.glob(test_dir+"\\*"+image_format)
else:
    for i in train_categories:
        training_filenames.append(glob.glob(train_dir+"/"+i+"/*"+image_format))

    test_filenames = glob.glob(test_dir+"/*"+image_format)

print("Number of Training Images: ",len(training_filenames[0])+len(training_filenames[1]))
print("Number of Testing Images: ",len(test_filenames))

training_labels=[]

test_labels=[]

test_labels = [0]*len(test_filenames) # won't be used anyway

for file in training_filenames[0]:
    training_labels.append(0) #mountain bikes

for file in training_filenames[1]:
    training_labels.append(1) #road bikes

#flattening it.
training_filenames = [val for sublist in training_filenames for val in sublist]
#Sanity Checking
#print(len(training_filenames))
#print(len(training_labels))

def process_function(filename,label):
    image_string = tf.read_file(filename)
    image_read = tf.image.decode_jpeg(image_string,channels=3)
    image_read=tf.image.convert_image_dtype(image_read,tf.float32)
    image_resized = tf.image.resize_images(image_read, [256,256])
    return image_resized, label

def train_preprocess(image, label):
    
    image = tf.image.per_image_standardization(image)
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


#Using TF dataset API
#Training Dataset
train_dataset=tf.data.Dataset.from_tensor_slices((training_filenames,training_labels))
#train_dataset=tf.data.Dataset.zip((training_filenames,training_labels))
train_dataset=train_dataset.shuffle(len(training_filenames))
train_dataset=train_dataset.map(process_function,num_parallel_calls=num_parallel_processors)
train_dataset=train_dataset.map(train_preprocess,num_parallel_calls=num_parallel_processors)
train_dataset=train_dataset.batch(batch_size)
train_dataset=train_dataset.repeat()
train_dataset=train_dataset.prefetch(5)

#Testing Dataset
test_dataset=tf.data.Dataset.from_tensor_slices((test_filenames,test_labels))
test_dataset=test_dataset.map(process_function,num_parallel_calls=num_parallel_processors)
test_dataset=test_dataset.batch(len(test_filenames))
test_dataset=test_dataset.prefetch(1)

# create general iterator
#print(train_dataset.output_types)
#print(train_dataset.output_shapes)

#test_iterator=test_dataset.make_initializable_iterator()
#im=test_iterator.get_next()
#testing_init_op=test_iterator.initializer

iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
training_init_op=iterator.make_initializer(train_dataset)
test_init_op=iterator.make_initializer(test_dataset)

#iterator=train_dataset.make_initializable_iterator()
images,labels = iterator.get_next()
#training_init_op = iterator.initializer
#inputs={'images':images,'labels':labels,'iterator_init_op': training_init_op}
inputs={'images':images,'labels':labels}

#defining the model
def simple_model(inputs):
    out=inputs["images"]
    out=tf.layers.conv2d(out,16,3,padding='same')
    out=tf.nn.relu(out)
    out=tf.layers.max_pooling2d(out,2,2)
    out=tf.reshape(out,[-1,128*128*16])
    out_logits=tf.layers.dense(out,num_classes)
    return out_logits

# create the neural network model
logits = simple_model(inputs)
labels=tf.cast(inputs["labels"],tf.int64)
# add the optimizer and loss
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op=optimizer.minimize(loss) ##IMPORTANT

#accuracy
prediction=tf.equal(tf.argmax(logits,1),labels)
prediction=tf.cast(prediction,tf.float32)
pred_prob = tf.reduce_sum(tf.nn.softmax(logits),1)
accuracy=tf.reduce_mean(prediction)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)  # initialize all variables
    sess.run(training_init_op)
    for i in range(epochs):
        _,loss_val,acc_val=sess.run([train_op,loss,accuracy])
        if i%50==0:
            print("Epoch number:{}, Loss:{:.3f}, Accuracy:{:.2f}".format(i,loss_val,acc_val))

    print("Training Done")
    sess.run(test_init_op)
    pred,prob=sess.run([logits,pred_prob])
    #For Debugging
    #print("Actual predictions",pred)
    pred_classes=np.argmax(pred, 1)
    print("Testing Done!")
    print("Class 0: %s, Class 1: %s" % (train_categories[0],train_categories[1]))
    print("Predicted Classes for each test image: ",pred_classes )
    #for probability 
    #print("Probability of class:", prob) 
