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
tf.flags.DEFINE_integer("max_itrs", "10000", "maximum iterations for training")
tf.flags.DEFINE_integer("batch_size", "128", "batch size for training")
tf.flags.DEFINE_integer("z_dim", "62", "size of input vector to generator")
tf.flags.DEFINE_integer("cd_dim", "10", "size of discrete code")
tf.flags.DEFINE_integer("cc_dim", "2", "size of continuous code")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Adam Optimizer")
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
      tf.get_variable('e_conv_0', shape = [4, 4, FLAGS.channel, ch_size], initializer=init_with_normal()),
      tf.get_variable('e_conv_1', shape = [4, 4, ch_size, ch_size*2], initializer=init_with_normal()),
      ]

  ch_size = FLAGS.d_ch_size
  last_layer_w = FLAGS.img_size//4 # => 28/(2**2) ->7

  WFCS = tf.get_variable('e_fc_shared', shape = [(ch_size*2)*last_layer_w*last_layer_w, 1024], initializer=init_with_normal())

  WY = tf.get_variable('e_fc_y', shape = [1024, 1], initializer=init_with_normal())

  WC = {
      'FC':tf.get_variable('e_fc_0', shape = [1024, 128], initializer=init_with_normal()),
      'CD':tf.get_variable('e_fc_cd', shape = [128, FLAGS.cd_dim], initializer=init_with_normal()),
      'CC':tf.get_variable('e_fc_cc', shape = [128, FLAGS.cc_dim], initializer=init_with_normal()),
      }

  return WEs, WFCS, WY, WC

def disc_model(x, WEs, WFCS, WY, WC, reuse):
  def batch_normalization(tensor):
    mean, var = tf.nn.moments(tensor, [0, 1, 2])
    out = tf.nn.batch_normalization(tensor, mean, var, 0, 1, FLAGS.eps)
    return out

  def leaky_relu(tensor):
    return tf.maximum(tensor*0.2, tensor)

  # encoder
  conved = tf.nn.conv2d(x, WEs[0], strides=[1, 2, 2, 1], padding='SAME')
  relued = leaky_relu(conved)

  conved = tf.nn.conv2d(relued, WEs[1], strides=[1, 2, 2, 1], padding='SAME')
  normalized = batch_norm_layer(conved, "discriminator/bne1", reuse)
  relued = leaky_relu(normalized)


  # flat 1-d vectors
  ch_size = FLAGS.d_ch_size
  last_layer_w = FLAGS.img_size//4 # => 28/(2**2) ->7
  reshaped = tf.reshape(relued, [-1, (ch_size*2)*last_layer_w*last_layer_w])

  # fully connected layer
  projected = tf.matmul(reshaped, WFCS)
  normalized = batch_norm_layer(projected, "discriminator/bne2", reuse)
  shared = leaky_relu(normalized)

  y = tf.matmul(shared, WY)

  cfc = tf.matmul(shared, WC['FC'])
  normalized = batch_norm_layer(cfc, "discriminator/bne3", reuse)
  relued = leaky_relu(normalized)

  cd = tf.matmul(relued, WC['CD'])

  cc = tf.matmul(relued, WC['CC'])

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


  # initialize weights, biases for Generator
  # shape=[kernel_size, kernel_size, (!)out_ch_size, (!)in_ch_size] for conv2d_transposed
  kernel_size = 4 
  WGs = [
      tf.get_variable('g_conv_0', shape = [kernel_size, kernel_size, 64, 128], initializer=init_with_normal()),
      tf.get_variable('g_conv_1', shape = [kernel_size, kernel_size, 1, 64], initializer=init_with_normal()),
      ]

  return WPJ, WGs

def gen_model(z_vecs, WPJ, WGs):
  def batch_normalization(tensor):
    mean, var = tf.nn.moments(tensor, [0, 1, 2])
    out = tf.nn.batch_normalization(tensor, mean, var, 0, 1, FLAGS.eps)
    return out

  img_size = FLAGS.img_size
  ch_size = FLAGS.g_ch_size
  batch_size = FLAGS.batch_size
  first_layer_w = FLAGS.img_size//4

  projected = tf.matmul(z_vecs, WPJ[0])
  normalized = batch_norm_layer(projected, "generator/bnpj0", False)
  relued = tf.nn.relu(normalized)
  
  projected = tf.matmul(relued, WPJ[1])
  reshaped = tf.reshape(projected, [-1, first_layer_w, first_layer_w, 128])
  normalized = batch_norm_layer(reshaped, "generator/bnpj1", False)
  relued = tf.nn.relu(normalized)

  deconved = tf.nn.conv2d_transpose(relued, WGs[0], [batch_size, 14, 14, 64], strides=[1, 2, 2, 1])
  normalized = batch_norm_layer(deconved, "generator/bng0", False)
  relued = tf.nn.relu(normalized)

  deconved = tf.nn.conv2d_transpose(relued, WGs[1], [batch_size, 28, 28, FLAGS.channel], strides=[1, 2, 2, 1])
  # skip batch normalization by DCGAN
  relued = tf.nn.relu(deconved)

  # no tanh
  contrastive_samples = relued
  return contrastive_samples

def get_regularization(cd, cc, cd_samples, cc_samples):

  cd_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cd, labels=cd_samples))
  cc_cross_entropy = tf.reduce_sum(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cc, labels=cc_samples), 0))
  
  return cd_cross_entropy + cc_cross_entropy

def get_opt(loss_val, scope):
  var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

  print "============================"
  print scope
  for item in var_list:
    print item.name

  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)
  grads = optimizer.compute_gradients(loss_val, var_list=var_list)
  return optimizer.apply_gradients(grads)

def preprocess(x):
  return x

def img_listup(img1, img2):
  h = img1.shape[0]
  w = img1.shape[1]
  out = np.zeros((h, w*2), np.uint8)
  out[:, :64] = img1
  out[:,64: ] = img2
  return out

def convert_img(data):
  out = cv2.resize(255* data.reshape(FLAGS.img_size, FLAGS.img_size), (64,64), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
  return out

def main(args):
  opts, args = getopt.getopt(sys.argv[1:], "s:", ["save_dir="])

  save_dir=FLAGS.save_dir

  for o, arg in opts:
    if o in ("-s", "--save_dir"):
      save_dir=arg
      print "checkpoint dir:", save_dir

  mnist = read_data_sets("./train_mnist")

  # setup for noise and code sample
  z_samples = tf.random_uniform([FLAGS.batch_size, FLAGS.z_dim], minval= -1.0, maxval=1.0)
  
  cd_samples_ = tf.reshape(tf.multinomial([[1.0]*FLAGS.cd_dim], FLAGS.batch_size), (-1, 1))
  onehot = tf.constant(np.eye(FLAGS.cd_dim, dtype=np.float32))
  cd_samples = tf.reshape(tf.nn.embedding_lookup(onehot, cd_samples_), (-1, FLAGS.cd_dim))
  
  cc_samples = tf.random_uniform([FLAGS.batch_size, FLAGS.cc_dim], minval = -1.0, maxval=1.0)
  
  z_vecs = tf.concat(axis=1, values=[z_samples, cd_samples, cc_samples])

  samples = tf.placeholder(tf.float32, [None, FLAGS.img_size, FLAGS.img_size, 1])

  with tf.variable_scope("discriminator") as scope:
    WEs, WFCS, WY, WC = init_disc_weights()

  with tf.variable_scope("generator") as scope:
    WPJ, WGs = init_gen_weights()

  logits_data, _, _ = disc_model(preprocess(samples), WEs, WFCS, WY, WC, False)
  cost_sample = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_data, labels=tf.constant(1.0, shape=[FLAGS.batch_size, 1])))
  contrastive_samples = gen_model(z_vecs, WPJ, WGs)
  logits_contrastive, contrastive_cd, contrastive_cc = disc_model(contrastive_samples, WEs, WFCS, WY, WC, True)
  negative_cost_contrastive = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_contrastive, labels=tf.constant(0.0, shape=[FLAGS.batch_size, 1])))

  cost_contrastive = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_contrastive, labels=tf.constant(1.0, shape=[FLAGS.batch_size, 1])))

  reg_loss = get_regularization(contrastive_cd, contrastive_cc, cd_samples, cc_samples)
  loss_d = cost_sample + negative_cost_contrastive + reg_loss
  loss_g = cost_contrastive + reg_loss

  disc_opt = get_opt(loss_d, "discriminator")
  gen_opt = get_opt(loss_g, "generator")

  points_data = tf.sigmoid(logits_data)
  points_contrastive = tf.sigmoid(logits_contrastive)
  
  start = datetime.now()
  print "Start: ",  start.strftime("%Y-%m-%d_%H-%M-%S")
  num_threads = FLAGS.num_threads
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=num_threads)) as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(save_dir)
    print "checkpoint: %s" % checkpoint
    if checkpoint:
      print "Restoring from checkpoint", checkpoint
      saver.restore(sess, checkpoint)
    else:
      print "Couldn't find checkpoint to restore from. Starting over."
      dt = datetime.now()
      filename = "checkpoint" + dt.strftime("%Y-%m-%d_%H-%M-%S")
      checkpoint = os.path.join(save_dir, filename)

    try:
      for itr in range(FLAGS.max_itrs):

        batch_data, _ = mnist.train.next_batch(FLAGS.batch_size)
        batch_data = batch_data.reshape(-1, FLAGS.img_size, FLAGS.img_size, 1)

        feed_dict = {samples: batch_data}

        print "------------------------------------------------------"
        print "[%05d]" % itr

        _ = sess.run([disc_opt], feed_dict=feed_dict)
        cost_sample_val, points_data_val = sess.run([cost_sample, points_data], feed_dict=feed_dict)
        print "\tcost_sample=", cost_sample_val, "points_data[0]:", points_data_val[0]

        _, cost_contrastive_val, points_contrastive_val = sess.run([gen_opt, cost_contrastive, points_contrastive], feed_dict=feed_dict)
        _, cost_contrastive_val, points_contrastive_val = sess.run([gen_opt, cost_contrastive, points_contrastive], feed_dict=feed_dict)

        print "\tcost_contrastive=", cost_contrastive_val, "points_contrastive[0]:",points_contrastive_val[0]

        sample_val = sess.run([samples[0]], feed_dict=feed_dict)
        cd_val, cc_val, contrastive_sample_val = sess.run([cd_samples[0], cc_samples[0],  contrastive_samples[0]], feed_dict=feed_dict)
        print "cd", cd_val
        print "cc", cc_val

        current = datetime.now()
        print "\telapsed:", current - start

        sample_vis = convert_img(batch_data[0])
        contrastive_sample_vis = convert_img(contrastive_sample_val)

        cv2.imshow('sample', img_listup(sample_vis, contrastive_sample_vis))
        cv2.waitKey(5)
        filepath = os.path.join(save_dir, "generated"+"%02d"%(itr%100) + "_%d"%(np.argmax(cd_val)) + ".png")
        
        scipy.misc.imsave(filepath, contrastive_sample_vis)
        if itr > 1 and itr % 300 == 0:
          #energy_d_val, loss_d_val, loss_g_val = sess.run([energy_d, loss_d, loss_g])
          print "#######################################################"
          #print "\tE=", energy_d_val, "Ld(x, z)=", loss_d, "Lg(z)=", loss_g
          saver.save(sess, checkpoint)

    except tf.errors.OutOfRangeError:
      print "the last epoch ends."

    cv2.destroyAllWindows()


if __name__ == "__main__":
  tf.app.run()
