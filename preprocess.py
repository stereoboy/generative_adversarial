import tensorflow as tf
import numpy as np
import glob
import os
import json

import setting

def crop_and_save(file_list, path):

  num_threads = 6
  batch_size = 6

  # reference: http://stackoverflow.com/questions/34783030/saving-image-files-in-tensorflow

  print type(file_list[0])
  #print file_list

  file_queue = tf.train.string_input_producer(file_list, shuffle=False)
  # crop data
  reader = tf.WholeFileReader()
  key, value = reader.read(file_queue)
  decoded = tf.image.decode_jpeg(value)
  cropped = tf.image.crop_to_bounding_box(decoded, 40, 20, 138, 138)
  resized = tf.cast(tf.image.resize_images(cropped, [setting.img_size, setting.img_size]), tf.uint8)
  encoded = tf.image.encode_jpeg(resized)

  encoded_list = tf.train.batch([encoded],
                                       batch_size=batch_size,
                                       num_threads=num_threads
                                       )

  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=num_threads)) as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    sess.run(tf.global_variables_initializer())

    for batch_i in range(0, (len(file_list) + batch_size - 1)/batch_size):
      resized_result = sess.run(encoded_list)
      print "=================================="
      print len(resized_result)
      print path
      for i in range(0, len(resized_result)):
        index = batch_i*batch_size + i
        if index < len(file_list):
          filename = os.path.join(path, os.path.basename(file_list[index]))
          print "save to:",filename
          with open(filename, "wb+") as save_file:
            save_file.write(resized_result[i])

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
  path = '/home/rofox/work/tensorflow/data/img_align_celeba/'
  #path = '/home/rofox/work/tensorflow/my_sample/my_ebgan/small_data'

  if not os.path.isfile("file_list.json"):
    trainimglist = glob.glob(path + '/*.jpg')

    print trainimglist[:10]

    num_list = len(trainimglist)
    offset = 0
    size = int(num_list*setting.validation_ratio)
    validation_file_list = trainimglist[offset: offset + size]

    offset = offset + size
    size = int(num_list*setting.test_ratio)
    test_file_list = trainimglist[offset: offset + size]

    offset = offset + size
    train_file_list = trainimglist[offset:]

    crop_and_save(train_file_list, "./train")
    crop_and_save(validation_file_list, "./validation")
    crop_and_save(test_file_list, "./test")

    print "train file size:", len(train_file_list)
    print "validation file size:", len(validation_file_list)
    print "test file size:", len(test_file_list)

    train_file_list      = map(lambda x: os.path.join("./train", os.path.basename(x)), train_file_list)
    validation_file_list = map(lambda x: os.path.join("./validation", os.path.basename(x)), validation_file_list)
    test_file_list       = map(lambda x: os.path.join("./test", os.path.basename(x)), test_file_list)

    with open("file_list.json", "wb+") as out:
      data = {'train':train_file_list, 'valid': validation_file_list, 'test': test_file_list}
      json.dump(data, out)


  with open("file_list.json") as in_file:
    data = json.load(in_file)
    print "=[check]====="
    print "train_file_list:", len(data['train'])
    print "validation_file_list:", len(data['valid'])
    print "test_file_list:", len(data['test'])
    train_file_list       = data['train']
    validation_file_list  = data['valid']
    test_file_list        = data['test']


