import numpy as np
import tensorflow as tf
import glob
import os
from PIL import Image

IMAGE_PATH = "/Users/wei/Pictures/cam/32x32/"
BATCH_SIZE = 100
N_EPOCHS = 1000

def labeled_image_list():
    labelfile = "/Users/wei/workspace/onoff-recognizer/labels.txt"

    with open(labelfile) as file:
        lines = [l.strip().split(",") for l in file]
    files = [e[0] for e in lines]
    labels = [[1.0,0.0] if e[1]=='0' else [0.0,1.0] for e in lines] # one-hot

    return files, labels

def read_images_from_disk(input_queue):
    """
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(IMAGE_PATH + input_queue[0])
    example = tf.reshape(tf.image.decode_png(file_contents, channels=3), [3*1024])
    return example, label

def readall(files):
    return np.array([np.array(Image.open(IMAGE_PATH+f).convert('RGB').getdata()).reshape(3072) for f in files])

# print(labeled_image_list())

image_list, label_list = labeled_image_list()
img_all = readall(image_list)

images = tf.convert_to_tensor(image_list, dtype=tf.string)
labels = tf.convert_to_tensor(label_list, dtype=tf.float32)

# Makes an input queue
input_queue = tf.train.slice_input_producer([images, labels], num_epochs=N_EPOCHS)

image, label = read_images_from_disk(input_queue)
train_image_batch, train_label_batch = tf.train.batch(
                                    [image, label],
                                    batch_size=BATCH_SIZE)

x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.float32, [None, 2])

M = 3700
W = tf.Variable(tf.truncated_normal([3072, M], stddev=0.1))
b = tf.Variable(tf.truncated_normal([M], stddev=0.1))
L1 = tf.nn.tanh(tf.matmul(x, W)+b)

N = 3450
W2 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
b2 = tf.Variable(tf.zeros([N]))
L2 = tf.nn.tanh(tf.matmul(L1, W2)+b2)

P = 3150
W3 = tf.Variable(tf.truncated_normal([N, P], stddev=0.1))
b3 = tf.Variable(tf.zeros([P]))
L3 = tf.nn.tanh(tf.matmul(L2, W3)+b3)

Wo = tf.Variable(tf.truncated_normal([P, 2], stddev=0.1))
bo = tf.Variable(tf.zeros([2]))

prediction = tf.nn.softmax(tf.matmul(L3, Wo)+bo)

# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)
train_step = tf.train.AdagradOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        step = 0
        while not coord.should_stop():
            dict = sess.run({x:train_image_batch, y:train_label_batch})
            sess.run(train_step, feed_dict=dict)

            if step % 10 == 0:
                train_acc = sess.run(accuracy, feed_dict={x:img_all, y:label_list})
                print("step(%d), Train acc: %f" % (step, train_acc))

            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training')
    finally:
        coord.request_stop()

     # Wait for threads to finish.
    coord.join(threads)
    sess.close()
