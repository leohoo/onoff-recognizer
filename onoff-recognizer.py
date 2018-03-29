import numpy as np
import tensorflow as tf
import glob
import os

def labeled_image_list():
    path = "~/Pictures/cam/32x32"

    files = []
    labels = []

    for sub in ["on", "off"]:
        p = os.path.expanduser(path + "/" + sub + "/*.png")
        list = glob.glob(p)
        files += list
        flags = [1 if "on"==sub else 0 for _ in list]
        labels += flags

    return files, labels

def read_images_from_disk(input_queue):
    """
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    return example, label

# print(labeled_image_list())

image_list, label_list = labeled_image_list()

images = tf.convert_to_tensor(image_list, dtype=tf.string)
labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

# Makes an input queue
input_queue = tf.train.slice_input_producer([images, labels])

image, label = read_images_from_disk(input_queue)
print(label)

with tf.Session() as sess:
    # sess.run(tf.global_variable_initializer())
    print(sess.run(label))
