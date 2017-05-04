import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import scipy.misc
import numpy as np
import glob
import os
import json
from datetime import datetime, date, time
import cv2
import sys
import getopt
import random

############################################################
#
# reference:
#  * https://github.com/openai/InfoGAN.git
#    * infogan related logic
#  * https://github.com/Newmu/dcgan_code.git
#  * https://github.com/soumith/dcgan.torch.git
#    * Generator Architecture for DCGAN
#  * https://github.com/shekkizh/EBGAN.tensorflow.git
#    * pull-away regularization term
#    * optimizer setup correspoding variable scope
############################################################

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("channel", "1", "batch size for training")
tf.flags.DEFINE_integer("max_epoch", "100", "maximum iterations for training")
tf.flags.DEFINE_integer("batch_size", "128", "batch size for training")
tf.flags.DEFINE_integer("z_dim", "62", "size of input vector to generator")
tf.flags.DEFINE_integer("cd_dim", "10", "size of discrete code")
tf.flags.DEFINE_integer("cc_dim", "2", "size of continuous code")
tf.flags.DEFINE_float("lambda0", "1.00", "lambda for Regularization Term")
tf.flags.DEFINE_float("learning_rate_D", "2e-4", "Learning rate for Adam Optimizer")
#tf.flags.DEFINE_float("learning_rate_G", "1e-3", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("learning_rate_G", "2e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("eps", "1e-5", "epsilon for various operation")
tf.flags.DEFINE_float("beta1", "0.5", "beta1 for Adam optimizer")
tf.flags.DEFINE_float("pt_w", "0.1", "weight of pull-away term")
tf.flags.DEFINE_float("margin", "20", "Margin to converge to for discriminator")
tf.flags.DEFINE_string("noise_type", "uniform", "noise type for z vectors")
tf.flags.DEFINE_string("save_dir", "info_mnist_checkpoints", "dir for checkpoints")
tf.flags.DEFINE_integer("img_size", "28", "sample image size")
tf.flags.DEFINE_integer("d_ch_size", "64", "channel size in last discriminator layer")
tf.flags.DEFINE_integer("g_ch_size", "128", "channel size in last generator layer")
tf.flags.DEFINE_integer("num_threads", "6", "max thread number")

def batch_norm_layer(tensors ,scope_bn, reuse):
  out = tf.contrib.layers.batch_norm(tensors, decay=0.9, center=True, scale=True,
      epsilon=FLAGS.eps,
      updates_collections=None,
      is_training=True,
      reuse=reuse,
      trainable=True,
      scope=scope_bn)
  return out

def init_disc_weights():
  def init_with_normal():
    return tf.truncated_normal_initializer(mean=0.0, stddev=0.02)

  ch_size = FLAGS.d_ch_size
  # initialize weights, biases for Encoder
  WEs = [
      tf.get_variable('e_conv_0', shape = [5, 5, FLAGS.channel, ch_size], initializer=init_with_normal()),
      tf.get_variable('e_conv_1', shape = [5, 5, ch_size, ch_size*2], initializer=init_with_normal()),
      ]
  
  BEs = [
      tf.get_variable('e_bias_0', shape = [ch_size], initializer=tf.zeros_initializer()),
      tf.get_variable('e_bias_1', shape = [ch_size*2], initializer=tf.zeros_initializer()),
      ]

  ch_size = FLAGS.d_ch_size
  last_layer_w = FLAGS.img_size//4 # => 28/(2**2) ->7

  WFCS = tf.get_variable('e_fc_shared', shape = [(ch_size*2)*last_layer_w*last_layer_w, 1024], initializer=init_with_normal())
  BFCS = tf.get_variable('e_biasfc_shared', shape = [1024], initializer=tf.zeros_initializer())

  WY = tf.get_variable('e_fc_y', shape = [1024, 1], initializer=init_with_normal())
  BY = tf.get_variable('e_biasfc_y', shape = [1], initializer=tf.zeros_initializer())

  WC = {
      'FC':tf.get_variable('e_fc_0', shape = [1024, 128], initializer=init_with_normal()),
      'CD':tf.get_variable('e_fc_cd', shape = [128, FLAGS.cd_dim], initializer=init_with_normal()),
      'CC':tf.get_variable('e_fc_cc', shape = [128, FLAGS.cc_dim], initializer=init_with_normal()),
      }
  BC = {
      'FC':tf.get_variable('e_biasfc_0', shape = [128], initializer=tf.zeros_initializer()),
      'CD':tf.get_variable('e_biasfc_cd', shape = [FLAGS.cd_dim], initializer=tf.zeros_initializer()),
      'CC':tf.get_variable('e_biasfc_cc', shape = [FLAGS.cc_dim], initializer=tf.zeros_initializer()),
      }

  return WEs, BEs, WFCS, BFCS, WY, BY, WC, BC

def disc_model(x, WEs, BEs, WFCS, BFCS, WY, BY, WC, BC, reuse):

  def leaky_relu(tensor):
    return tf.maximum(tensor*0.2, tensor)

  # encoder
  conved = tf.nn.conv2d(x, WEs[0], strides=[1, 2, 2, 1], padding='SAME')
  conved = tf.nn.bias_add(conved, BEs[0])
  relued = leaky_relu(conved)

  conved = tf.nn.conv2d(relued, WEs[1], strides=[1, 2, 2, 1], padding='SAME')
  conved = tf.nn.bias_add(conved, BEs[1])
  normalized = batch_norm_layer(conved, "discriminator/bne1", reuse)
  relued = leaky_relu(normalized)


  # flat 1-d vectors
  ch_size = FLAGS.d_ch_size
  last_layer_w = FLAGS.img_size//4 # => 28/(2**2) ->7
  reshaped = tf.reshape(relued, [-1, (ch_size*2)*last_layer_w*last_layer_w])

  # fully connected layer
  projected = tf.matmul(reshaped, WFCS)
  projected = tf.nn.bias_add(projected, BFCS)
  normalized = batch_norm_layer(projected, "discriminator/bne2", reuse)
  shared = leaky_relu(normalized)

  y = tf.matmul(shared, WY)
  y = tf.nn.bias_add(y, BY)

  cfc = tf.matmul(shared, WC['FC'])
  cfc = tf.nn.bias_add(cfc, BC['FC'])
  normalized = batch_norm_layer(cfc, "discriminator/bne3", reuse)
  relued = leaky_relu(normalized)

  cd = tf.matmul(relued, WC['CD'])
  cd = tf.nn.bias_add(cd, BC['CD'])

  cc = tf.matmul(relued, WC['CC'])
  cc = tf.nn.bias_add(cc, BC['CC'])

  return y, cd, cc

def init_gen_weights():
  def init_with_normal():
    return tf.truncated_normal_initializer(mean=0.0, stddev=0.02)

  ch_size = FLAGS.g_ch_size

  # initialize weights, biases for Projection from Z to the first layer in generator
  dim = FLAGS.z_dim + FLAGS.cd_dim + FLAGS.cc_dim
  first_layer_w = FLAGS.img_size//4
  WPJ = [
      tf.get_variable('g_proj_0', shape = [dim, 1024], initializer=init_with_normal()),
      tf.get_variable('g_proj_1', shape = [1024, first_layer_w*first_layer_w*128], initializer=init_with_normal()),
      ]
  
  BPJ = [
      tf.get_variable('g_biasproj_0', shape = [1024], initializer=tf.zeros_initializer()),
      tf.get_variable('g_biasproj_1', shape = [first_layer_w*first_layer_w*128], initializer=tf.zeros_initializer()),
      ]


  # initialize weights, biases for Generator
  # shape=[kernel_size, kernel_size, (!)out_ch_size, (!)in_ch_size] for conv2d_transposed
  kernel_size = 5
  WGs = [
      tf.get_variable('g_conv_0', shape = [kernel_size, kernel_size, 64, 128], initializer=init_with_normal()),
      tf.get_variable('g_conv_1', shape = [kernel_size, kernel_size, 1, 64], initializer=init_with_normal()),
      ]
  BGs = [
      tf.get_variable('g_bias_0', shape = [64], initializer=tf.zeros_initializer()),
      tf.get_variable('g_bias_1', shape = [1], initializer=tf.zeros_initializer()),
      ]

  return WPJ, BPJ, WGs, BGs

def gen_model(z_vecs, WPJ, BPJ, WGs, BGs):
  img_size = FLAGS.img_size
  ch_size = FLAGS.g_ch_size
  batch_size = FLAGS.batch_size
  first_layer_w = FLAGS.img_size//4

  projected = tf.matmul(z_vecs, WPJ[0])
  projected = tf.nn.bias_add(projected, BPJ[0])
  normalized = batch_norm_layer(projected, "generator/bnpj0", False)
  relued = tf.nn.relu(normalized)

  projected = tf.matmul(relued, WPJ[1])
  projected = tf.nn.bias_add(projected, BPJ[1])
  reshaped = tf.reshape(projected, [-1, first_layer_w, first_layer_w, 128])
  normalized = batch_norm_layer(reshaped, "generator/bnpj1", False)
  relued = tf.nn.relu(normalized)

  deconved = tf.nn.conv2d_transpose(relued, WGs[0], [batch_size, 14, 14, 64], strides=[1, 2, 2, 1])
  deconved = tf.nn.bias_add(deconved, BGs[0])
  normalized = batch_norm_layer(deconved, "generator/bng0", False)
  relued = tf.nn.relu(normalized)

  deconved = tf.nn.conv2d_transpose(relued, WGs[1], [batch_size, 28, 28, FLAGS.channel], strides=[1, 2, 2, 1])
  deconved = tf.nn.bias_add(deconved, BGs[1])
  # skip batch normalization by DCGAN
  #relued = tf.nn.relu(deconved)

  # no tanh
  #contrastive_samples = tf.clip_by_value(deconved, 0, 1.0)
  contrastive_samples = tf.sigmoid(deconved)
  #contrastive_samples = deconved
  return contrastive_samples

def get_regularization(cd, cc, cd_samples, cc_samples):
  # FIXME

  cd_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cd, labels=cd_samples))
  #cc_cross_entropy = tf.reduce_sum(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cc, labels=cc_samples), 0))

  ret = cd_cross_entropy
  #ret = cd_cross_entropy + cc_cross_entropy
  return  ret
def get_opt_D(loss_val, scope):
  var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

  print("============================")
  print(scope)
  for item in var_list:
    print(item.name)

  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate_D, beta1=FLAGS.beta1)
  grads = optimizer.compute_gradients(loss_val, var_list=var_list)
  return optimizer.apply_gradients(grads)

def get_opt_G(loss_val, scope):
  var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

  print("============================")
  print(scope)
  for item in var_list:
    print(item.name)

  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate_G, beta1=FLAGS.beta1)
  grads = optimizer.compute_gradients(loss_val, var_list=var_list)
  return optimizer.apply_gradients(grads)

def preprocess(x):
  return x

def img_listup(imgs):
  size = len(imgs)
  (h, w) = imgs[0].shape[:2]

  total_w = 0
  for img in imgs:
    total_w += img.shape[1]
  out = np.zeros((h, total_w), np.uint8)

  offset = 0
  for i in range(size):
    h, w = imgs[i].shape[:2]
    out[:h, offset: offset + w] = imgs[i]
    offset += w

  return out

def convert_img(data):
  np.set_printoptions(threshold=np.nan)

  #data = np.clip(data, 0.0, 1.0)
  out = cv2.resize(255* data.reshape(FLAGS.img_size, FLAGS.img_size), (56,56), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
  #out = cv2.resize(255* data.reshape(FLAGS.img_size, FLAGS.img_size), (56,56)).astype(np.uint8)
  #out = (255*data.reshape(FLAGS.img_size, FLAGS.img_size)).astype(np.uint8)
  #out = cv2.resize(255* data.reshape(FLAGS.img_size, FLAGS.img_size), (64,64)).astype(np.uint8)
  return out

def get_random_descrete_codes():
  cd_data = np.random.randint(0, FLAGS.cd_dim, size=(FLAGS.batch_size))
  return cd_data

def get_random_continuous_codes():
  cc_data = np.random.uniform(-1, 1, size=(FLAGS.batch_size, 2))
  return cc_data

def get_random_z():
  z_data = np.random.uniform(-1, 1, size=(FLAGS.batch_size, FLAGS.z_dim))
  return z_data

def get_random_codes():
  cd_data = get_random_descrete_codes()
  cc_data = get_random_continuous_codes()
  z_data = get_random_z()
  return cd_data, cc_data, z_data

def main(args):
  opts, args = getopt.getopt(sys.argv[1:], "s:", ["save_dir="])

  save_dir=FLAGS.save_dir

  for o, arg in opts:
    if o in ("-s", "--save_dir"):
      save_dir=arg
      print("checkpoint dir:" + save_dir)

  mnist = read_data_sets("./train_mnist")

  train_set = mnist.train.images
  val_set = mnist.validation.images
  test_set = mnist.test.images

  #train_set = np.concatenate([train_set, val_set, test_set], axis=0)
  #train_set = np.concatenate([train_set, test_set], axis=0)
  train_size = len(train_set)

  # setup for noise and code sample
  z_samples = tf.random_uniform([FLAGS.batch_size, FLAGS.z_dim], minval= -1.0, maxval=1.0)
  #z_samples = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.z_dim])

#  cd_samples_ = tf.reshape(tf.multinomial([[1.0]*FLAGS.cd_dim], FLAGS.batch_size), (-1, 1))
#  onehot = tf.constant(np.eye(FLAGS.cd_dim, dtype=np.float32))
#  cd_samples = tf.reshape(tf.nn.embedding_lookup(onehot, cd_samples_), (-1, FLAGS.cd_dim))
  _cd_samples = tf.placeholder(tf.int32, [FLAGS.batch_size])
  cd_samples = tf.one_hot(_cd_samples, FLAGS.cd_dim)

  #cc_samples = tf.random_uniform([FLAGS.batch_size, FLAGS.cc_dim], minval = -1.0, maxval=1.0)
  cc_samples = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.cc_dim])

  z_vecs = tf.concat(axis=1, values=[z_samples, cd_samples, cc_samples])

  samples = tf.placeholder(tf.float32, [None, FLAGS.img_size, FLAGS.img_size, 1])

  with tf.variable_scope("discriminator") as scope:
    WEs, BEs, WFCS, BFCS, WY, BY, WC, BC = init_disc_weights()

  with tf.variable_scope("generator") as scope:
    WPJ, BPJ, WGs, BGs = init_gen_weights()

  logits_data, _, _ = disc_model(preprocess(samples), WEs, BEs, WFCS, BFCS, WY, BY, WC, BC, False)
  cost_sample = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_data, labels=tf.constant(1.0, shape=[FLAGS.batch_size, 1])))

  contrastive_samples = gen_model(z_vecs, WPJ, BPJ, WGs, BGs)
  logits_contrastive, contrastive_cd, contrastive_cc = disc_model(contrastive_samples, WEs, BEs, WFCS, BFCS, WY, BY, WC, BC, True)
  negative_cost_contrastive = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_contrastive, labels=tf.constant(0.0, shape=[FLAGS.batch_size, 1])))

  cost_contrastive = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_contrastive, labels=tf.constant(1.0, shape=[FLAGS.batch_size, 1])))

  reg_loss = get_regularization(contrastive_cd, contrastive_cc, cd_samples, cc_samples)
  loss_d = cost_sample + negative_cost_contrastive + FLAGS.lambda0*reg_loss
  loss_g = cost_contrastive + FLAGS.lambda0*reg_loss

  disc_opt = get_opt_D(loss_d, "discriminator")
  gen_opt = get_opt_G(loss_g, "generator")

  points_data = tf.reduce_mean(tf.sigmoid(logits_data))
  points_contrastive = tf.reduce_mean(tf.sigmoid(logits_contrastive))

  start = datetime.now()
  print("Start: " +  start.strftime("%Y-%m-%d_%H-%M-%S"))
  num_threads = FLAGS.num_threads
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=num_threads)) as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(save_dir)
    print("checkpoint: %s" % checkpoint)
    if checkpoint:
      print("Restoring from checkpoint %s"%checkpoint)
      saver.restore(sess, checkpoint)
    else:
      print("Couldn't find checkpoint to restore from. Starting over.")
      dt = datetime.now()
      filename = "checkpoint" + dt.strftime("%Y-%m-%d_%H-%M-%S")
      checkpoint = os.path.join(save_dir, filename)

    val_cost_sample_val = 1.0
    cost_contrastive_val = 0
    max_itr = train_size//FLAGS.batch_size
    for epoch in range(FLAGS.max_epoch):
      #random.shuffle(train_set)
      print("#####################################################################")
      for itr in range(0, max_itr):
      #for itr in range(0, 0):

        begin = itr*FLAGS.batch_size
        end = (itr + 1)*FLAGS.batch_size
        batch_data = train_set[begin:end]
        batch_data = batch_data.reshape(-1, FLAGS.img_size, FLAGS.img_size, 1)

        cd_data, cc_data, z_data = get_random_codes()
        feed_dict = {samples: batch_data, _cd_samples: cd_data, cc_samples: cc_data}
        _, cost_sample_val, points_data_val = sess.run([disc_opt, cost_sample, points_data], feed_dict=feed_dict)
        #cost_sample_val, points_data_val = sess.run([cost_sample, points_data], feed_dict=feed_dict)

        cd_data, cc_data, z_data = get_random_codes()
        feed_dict = {samples: batch_data, _cd_samples: cd_data, cc_samples: cc_data}
        sess.run(gen_opt, feed_dict=feed_dict)
        
        cd_data, cc_data, z_data = get_random_codes()
        feed_dict = {samples: batch_data, _cd_samples: cd_data, cc_samples: cc_data}
        _, cost_contrastive_val, points_contrastive_val = sess.run([gen_opt, cost_contrastive, points_contrastive], feed_dict=feed_dict)
        while epoch > 0 and points_contrastive_val < 0.40:
          print("-", end='')
          sess.run(gen_opt, feed_dict=feed_dict)
          _, cost_contrastive_val, points_contrastive_val = sess.run([gen_opt, cost_contrastive, points_contrastive], feed_dict=feed_dict)

        if itr > 1 and itr % 10 == 0:
          print("")
          print("===================================================================")
          print("[%d] %03d/%d"%(epoch, itr, max_itr))

          #cost_sample_val, points_data_val = sess.run([cost_sample, points_data], feed_dict=feed_dict)
          print("\tcost_sample=%f points_data[0]:%f"%(cost_sample_val,  points_data_val))

          #cost_contrastive_val, points_contrastive_val = sess.run([cost_contrastive, points_contrastive], feed_dict=feed_dict)
          print("\tcost_contrastive=%f points_contrastive[0]:%f"%(cost_contrastive_val, points_contrastive_val))

          cv2.imshow('input', cv2.resize(img_listup(255*np.squeeze(batch_data[:20])), (56*20, 56), cv2.INTER_NEAREST))

          cd_data, cc_data, z_data = get_random_codes()
          cd_data = np.array(list(range(FLAGS.batch_size)), np.int32)%FLAGS.cd_dim
          feed_dict = {_cd_samples: cd_data, cc_samples: cc_data}
          cd_val, cc_val, contrastive_sample_val = sess.run([cd_samples, cc_samples,  contrastive_samples], feed_dict=feed_dict)
          print("\tcd:{}".format(cd_val[0]))
          print("\tcc:{}".format(cc_val[0]))

          current = datetime.now()
          print("\telapsed:", current - start)

          imgs = []
          for i in range(20):
            sample_vis = convert_img(batch_data[i])
            contrastive_sample_vis = convert_img(contrastive_sample_val[i])

            imgs.append(contrastive_sample_vis)
            filepath = os.path.join(save_dir, "generated" + "_%02d"%(epoch) + "_%d"%(np.argmax(cd_val[i])) + "_%02d"%(itr%100) + ".png")
            #filepath = os.path.join(save_dir, "generated" + "_%d"%(np.argmax(cd_val[i])) + "_%02d"%(itr%100) + ".png")

            scipy.misc.imsave(filepath, contrastive_sample_vis)
          vis_0 = img_listup(imgs[:10])
          cv2.imshow('sample 0', vis_0)
          filepath = os.path.join(save_dir, "vis" + "_%02d"%(epoch) + "_%02d"%(itr%100) + ".png")
          scipy.misc.imsave(filepath, vis_0)
         
          ################################################################
          cd_data, cc_data, z_data = get_random_codes()
          cd_data = np.full((FLAGS.batch_size), 0, np.int32)
          feed_dict = {_cd_samples: cd_data, cc_samples: cc_data}
          cd_val, cc_val, contrastive_sample_val = sess.run([cd_samples, cc_samples,  contrastive_samples], feed_dict=feed_dict)
          print("\tcd:{}".format(cd_val[0]))
          print("\tcc:{}".format(cc_val[0]))

          current = datetime.now()
          print("\telapsed:", current - start)

          imgs = []
          for i in range(20):
            sample_vis = convert_img(batch_data[i])
            contrastive_sample_vis = convert_img(contrastive_sample_val[i])

            imgs.append(contrastive_sample_vis)
            filepath = os.path.join(save_dir, "generated" + "_%02d"%(epoch) + "_%d"%(np.argmax(cd_val[i])) + "_%02d"%(itr%100) + ".png")
            #filepath = os.path.join(save_dir, "generated" + "_%d"%(np.argmax(cd_val[i])) + "_%02d"%(itr%100) + ".png")

            scipy.misc.imsave(filepath, contrastive_sample_vis)
          cv2.imshow('sample1', img_listup(imgs))
        cv2.waitKey(5)

      print("#######################################################")
#      # evaluate
#      total_cost_val = 0.0
#      max_val_itr = len(val_set)//FLAGS.batch_size
#      for itr in range(0, max_val_itr):
#        begin = itr*FLAGS.batch_size
#        end = (itr + 1)*FLAGS.batch_size
#        batch_data = val_set[begin:end]
#        batch_data = batch_data.reshape(-1, FLAGS.img_size, FLAGS.img_size, 1)
#
#        feed_dict = {samples: batch_data}
#        cost_sample_val, points_data_val = sess.run([cost_sample, points_data], feed_dict=feed_dict)
#        cost_sample_val = sess.run(cost_sample, feed_dict=feed_dict)
#        total_cost_val += cost_sample_val
#        val_cost_sample_val = total_cost_val/max_val_itr
#      print("Validation loss: %f"%(val_cost_sample_val))

      img_save_dir = FLAGS.save_dir + "/epoch_" + str(epoch)
      if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)

#      for cd in range(FLAGS.cd_dim):
#      #for cd in range(1):
#        cd_data = np.array(list(range(FLAGS.batch_size)), np.int32)%FLAGS.cd_dim
#        #cd_data = np.array([cd]*FLAGS.batch_size, np.int32)%FLAGS.cd_dim
#        cc_data = get_random_continuous_codes()
#        #feed_dict = {_cd_samples: cd_data, cc_samples: cc_data}
#        z_data = get_random_z()
#        feed_dict = {samples: batch_data, _cd_samples: cd_data, cc_samples: cc_data}
#        cd_val, cc_val, contrastive_sample_val = sess.run([cd_samples, cc_samples,  contrastive_samples], feed_dict=feed_dict)
#        imgs = []
#        for i in range(20):
#          contrastive_sample_vis = convert_img(contrastive_sample_val[i])
#          imgs.append(contrastive_sample_vis)
#          filepath = os.path.join(img_save_dir, "generated" + "_%d"%(np.argmax(cd_val[i])) + "_%02d"%(i) + ".png")
#          scipy.misc.imsave(filepath, contrastive_sample_vis)
#        cv2.imshow('class' + str(cd), img_listup(imgs))

      for i in range(11):
        #cc_data = get_random_continuous_codes()
        cd_data = np.array(list(range(FLAGS.batch_size)), np.int32)%FLAGS.cd_dim
        offset = 0.2
        cc_data = np.array([[-1.0 + i*offset, np.random.uniform(-1, 1)] for _ in range(FLAGS.batch_size)], np.float32)
        feed_dict = {_cd_samples: cd_data, cc_samples: cc_data}
        cd_val, cc_val, contrastive_sample_val = sess.run([cd_samples, cc_samples,  contrastive_samples], feed_dict=feed_dict)
        imgs = []
        for j in range(2*FLAGS.cd_dim):
          contrastive_sample_vis = convert_img(contrastive_sample_val[j])
          imgs.append(contrastive_sample_vis)
          filepath = os.path.join(img_save_dir, "generated" + "_%d"%(np.argmax(cd_val[j])) + "_%f"%(cc_val[j][0]) + ".png")
          scipy.misc.imsave(filepath, contrastive_sample_vis)
        cv2.imshow('class' + str(i), img_listup(imgs))

      saver.save(sess, checkpoint)


    cv2.destroyAllWindows()


if __name__ == "__main__":
  tf.app.run()
