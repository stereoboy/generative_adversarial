import tensorflow as tf
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
#  * https://github.com/Newmu/dcgan_code.git
#  * https://github.com/soumith/dcgan.torch.git
#    * Generator Architecture for DCGAN
#  * https://github.com/shekkizh/EBGAN.tensorflow.git
#    * pull-away regularization term
#    * optimizer setup correspoding variable scope
############################################################

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("channel", "3", "batch size for training")
tf.flags.DEFINE_integer("max_itrs", "10000", "maximum iterations for training")
tf.flags.DEFINE_integer("batch_size", "128", "batch size for training")
tf.flags.DEFINE_integer("z_dim", "100", "size of input vector to generator")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("eps", "1e-5", "epsilon for various operation")
tf.flags.DEFINE_float("beta1", "0.5", "beta1 for Adam optimizer")
tf.flags.DEFINE_float("pt_w", "0.1", "weight of pull-away term")
tf.flags.DEFINE_float("margin", "20", "Margin to converge to for discriminator")
tf.flags.DEFINE_string("noise_type", "uniform", "noise type for z vectors")
tf.flags.DEFINE_string("save_dir", "checkpoints", "dir for checkpoints")
tf.flags.DEFINE_integer("img_size", "64", "sample image size")
tf.flags.DEFINE_integer("g_ch_size", "64", "channel size in last discriminator layer")
tf.flags.DEFINE_integer("d_ch_size", "32", "channel size in last discriminator layer")
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
    return tf.truncated_normal_initializer(mean=0.0, stddev=0.002)

  ch_size = FLAGS.d_ch_size
  # initialize weights, biases for Encoder
  WEs = [
      tf.get_variable('e_conv_0', shape = [4, 4, FLAGS.channel, ch_size], initializer=init_with_normal()),
      tf.get_variable('e_conv_1', shape = [4, 4, ch_size, ch_size*2], initializer=init_with_normal()),
      tf.get_variable('e_conv_2', shape = [4, 4, ch_size*2, ch_size*4], initializer=init_with_normal()),
      ]

  BEs = [
      tf.get_variable('e_bias_0', shape = [ch_size], initializer=init_with_normal()),
      tf.get_variable('e_bias_1', shape = [ch_size*2], initializer=init_with_normal()),
      tf.get_variable('e_bias_2', shape = [ch_size*4], initializer=init_with_normal()),
      ]

  # initialize weights, biases for Decoder
  # shape=[kernel_size, kernel_size, out_ch_size, in_ch_size] for conv2d_transposed
  WDs = [
      tf.get_variable('d_conv_0', shape = [4, 4, ch_size*2, ch_size*4], initializer=init_with_normal()),
      tf.get_variable('d_conv_1', shape = [4, 4, ch_size, ch_size*2], initializer=init_with_normal()),
      tf.get_variable('d_conv_2', shape = [4, 4, FLAGS.channel, ch_size], initializer=init_with_normal()),
      ]

  BDs = [
      tf.get_variable('d_bias_0', shape = [ch_size*2], initializer=init_with_normal()),
      tf.get_variable('d_bias_1', shape = [ch_size], initializer=init_with_normal()),
      tf.get_variable('d_bias_2', shape = [FLAGS.channel], initializer=init_with_normal()),
      ]
  return WEs, BEs, WDs, BDs

def disc_model(x, WEs, BEs, WDs, BDs, reuse):
  def batch_normalization(tensor):
    mean, var = tf.nn.moments(tensor, [0, 1, 2])
    out = tf.nn.batch_normalization(tensor, mean, var, 0, 1, FLAGS.eps)
    return out

  def leaky_relu(tensor):
    return tf.maximum(tensor*0.2, tensor)

  # encoder
  conved = tf.nn.bias_add(tf.nn.conv2d(x, WEs[0], strides=[1, 2, 2, 1], padding='SAME'), BEs[0])
  # skip batch normalization by DCGAN
  #normalized = batch_normalization(conved)
  #relued = leaky_relu(normalized)
  relued = leaky_relu(conved)

  conved = tf.nn.bias_add(tf.nn.conv2d(relued, WEs[1], strides=[1, 2, 2, 1], padding='SAME'), BEs[1])
  #normalized = batch_normalization(conved)
  normalized = batch_norm_layer(conved, "discriminator/bne1", reuse)
  relued = leaky_relu(normalized)

  conved = tf.nn.bias_add(tf.nn.conv2d(relued, WEs[2], strides=[1, 2, 2, 1], padding='SAME'), BEs[2])
  #normalized = batch_normalization(conved)
  normalized = batch_norm_layer(conved, "discriminator/bne2", reuse)
  relued = leaky_relu(normalized)

  # project 1-d vectors
  ch_size = FLAGS.d_ch_size
  encoded_layer_w = FLAGS.img_size//8
  encoded = tf.reshape(relued, [-1, (ch_size*4)*encoded_layer_w*encoded_layer_w])

  # decoder
  batch_size = FLAGS.batch_size
  deconved = tf.nn.bias_add(tf.nn.conv2d_transpose(relued, WDs[0], [batch_size, 16, 16, ch_size*2], strides=[1, 2, 2, 1]), BDs[0])
  #normalized = batch_normalization(deconved)
  normalized = batch_norm_layer(deconved, "discriminator/bnd0", reuse)
  relued = leaky_relu(normalized)

  deconved = tf.nn.bias_add(tf.nn.conv2d_transpose(relued, WDs[1], [batch_size, 32, 32, ch_size], strides=[1, 2, 2, 1]), BDs[1])
  #normalized = batch_normalization(deconved)
  normalized = batch_norm_layer(deconved, "discriminator/bnd1", reuse)
  relued = leaky_relu(normalized)

  deconved = tf.nn.bias_add(tf.nn.conv2d_transpose(relued, WDs[2], [batch_size, 64, 64, FLAGS.channel], strides=[1, 2, 2, 1]), BDs[2])
  normalized = batch_normalization(deconved)
  decoded = tf.nn.tanh(normalized)

  # energy
  #loss = tf.sqrt(2 * tf.nn.l2_loss(x - decoded))
  loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(x - decoded), [1, 2, 3])))
  #loss = tf.sqrt(2 * tf.nn.l2_loss(x - decoded)) / FLAGS.batch_size
  return loss, encoded, decoded

def init_gen_weights():
  def init_with_normal():
    return tf.truncated_normal_initializer(mean=0.0, stddev=0.02)

  ch_size = FLAGS.g_ch_size

  # initialize weights, biases for Projection from Z to the first layer in generator
  WPJ = tf.get_variable('g_proj_w', shape = [FLAGS.z_dim, (ch_size*8)*4*4], initializer=init_with_normal())
  BPJ = tf.get_variable('g_proj_b', shape = [(ch_size*8)*4*4], initializer=init_with_normal())

  # initialize weights, biases for Generator
  # shape=[kernel_size, kernel_size, (!)out_ch_size, (!)in_ch_size] for conv2d_transposed
  WGs = [
      tf.get_variable('g_conv_0', shape = [4, 4, ch_size*4, ch_size*8], initializer=init_with_normal()),
      tf.get_variable('g_conv_1', shape = [4, 4, ch_size*2, ch_size*4], initializer=init_with_normal()),
      tf.get_variable('g_conv_2', shape = [4, 4, ch_size, ch_size*2], initializer=init_with_normal()),
      tf.get_variable('g_conv_3', shape = [4, 4, FLAGS.channel, ch_size], initializer=init_with_normal()),
      ]

  BGs = [
      tf.get_variable('g_bias_0', shape = [ch_size*4], initializer=init_with_normal()),
      tf.get_variable('g_bias_1', shape = [ch_size*2], initializer=init_with_normal()),
      tf.get_variable('g_bias_2', shape = [ch_size], initializer=init_with_normal()),
      tf.get_variable('g_bias_3', shape = [FLAGS.channel], initializer=init_with_normal()),
      ]

  return WPJ, BPJ, WGs, BGs

def gen_model(z_vecs, WPJ, BPJ, WGs, BGs):
  def batch_normalization(tensor):
    mean, var = tf.nn.moments(tensor, [0, 1, 2])
    out = tf.nn.batch_normalization(tensor, mean, var, 0, 1, FLAGS.eps)
    return out

  img_size = FLAGS.img_size
  ch_size = FLAGS.g_ch_size
  batch_size = FLAGS.batch_size

  projected = tf.matmul(z_vecs, WPJ) + BPJ
  reshaped = tf.reshape(projected, [-1, 4, 4, ch_size*8])
  #normalized = batch_normalization(reshaped)
  normalized = batch_norm_layer(reshaped, "generator/bnpj", False)
  relued = tf.nn.relu(normalized)

  deconved = tf.nn.bias_add(tf.nn.conv2d_transpose(relued, WGs[0], [batch_size, 8, 8, ch_size*4], strides=[1, 2, 2, 1]), BGs[0])
  #normalized = batch_normalization(deconved)
  normalized = batch_norm_layer(deconved, "generator/bng0", False)
  relued = tf.nn.relu(normalized)

  deconved = tf.nn.bias_add(tf.nn.conv2d_transpose(relued, WGs[1], [batch_size, 16, 16, ch_size*2], strides=[1, 2, 2, 1]), BGs[1])
  #normalized = batch_normalization(deconved)
  normalized = batch_norm_layer(deconved, "generator/bng1", False)
  relued = tf.nn.relu(normalized)

  deconved = tf.nn.bias_add(tf.nn.conv2d_transpose(relued, WGs[2], [batch_size, 32, 32, ch_size], strides=[1, 2, 2, 1]), BGs[2])
  #normalized = batch_normalization(deconved)
  normalized = batch_norm_layer(deconved, "generator/bng2", False)
  relued = tf.nn.relu(normalized)

  deconved = tf.nn.bias_add(tf.nn.conv2d_transpose(relued, WGs[3], [batch_size, 64, 64, FLAGS.channel], strides=[1, 2, 2, 1]), BGs[3])
  # skip batch normalization by DCGAN
  contrastive_samples = tf.nn.tanh(deconved)

  return contrastive_samples

def get_PT(encoded):
  norm = tf.sqrt(tf.reduce_sum(tf.square(encoded), 1, keep_dims=True)) # keep_dims=True to use this as a denominator
  normalized_embeddings = encoded / norm
  similarity = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)
  batch_size = FLAGS.batch_size
  pt_loss = 0.5*tf.reduce_sum(tf.square(similarity))/ (batch_size * (batch_size - 1))
  return pt_loss

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
  # map x into [-1, 1]
  return x/127.5 - 1

def get_samples(file_list):

  num_threads = 6

  # reference: http://stackoverflow.com/questions/34783030/saving-image-files-in-tensorflow

  print file_list[:10]
  print type(file_list)
  print FLAGS.batch_size

  file_queue = tf.train.string_input_producer(file_list, shuffle=False)
  reader = tf.WholeFileReader()
  key, value = reader.read(file_queue)
  decoded = tf.image.decode_jpeg(value)
  casted = tf.cast(decoded, tf.float32)

  samples = tf.train.shuffle_batch([casted],
                                   batch_size=FLAGS.batch_size,
                                   num_threads=num_threads,
                                   capacity=FLAGS.batch_size*200,
                                   min_after_dequeue=FLAGS.batch_size*100,
                                   shapes=[[FLAGS.img_size, FLAGS.img_size, FLAGS.channel]]
                                   )
  return samples

def img_listup(img1, img2):
  out = np.zeros((64, 64*2, 3), np.uint8)
  out[:, :64] = img1
  out[:,64: ] = img2
  return out

def convert_img(data):
  return tf.cast((data + 1.0)*127.5, tf.uint8)


def main(args):
  opts, args = getopt.getopt(sys.argv[1:], "s:", ["save_dir="])

  save_dir=FLAGS.save_dir

  for o, arg in opts:
    if o in ("-s", "--save_dir"):
      save_dir=arg
      print "checkpoint dir:", save_dir


  with open("file_list.json") as in_file:
    data = json.load(in_file)
    print "=[check]====="
    print "train_file_list:", len(data['train'])
    print "validation_file_list:", len(data['valid'])
    print "test_file_list:", len(data['test'])
    train_file_list       = data['train']
    validation_file_list  = data['valid']
    test_file_list        = data['test']

  samples = get_samples(train_file_list)
  z_vecs = tf.placeholder(tf.float32, [None, FLAGS.z_dim], name="z_vecs")

  with tf.variable_scope("discriminator") as scope:
    WEs, BEs, WDs, BDs = init_disc_weights()

  with tf.variable_scope("generator") as scope:
    WPJ, BPJ, WGs, BGs = init_gen_weights()


  energy_d, _, decoded = disc_model(preprocess(samples), WEs, BEs, WDs, BDs, False)

  contrastive_samples = gen_model(z_vecs, WPJ, BPJ, WGs, BGs)
  energy_g, contrastive_encoded, contrastive_decoded = disc_model(contrastive_samples, WEs, BEs, WDs, BDs, True)

  loss_d = energy_d + tf.maximum(0.0, FLAGS.margin - energy_g)
  #loss_d = energy_d +  FLAGS.margin - energy_g
  #loss_d = energy_d  - energy_g
  #loss_pt = FLAGS.pt_w*get_PT(contrastive_encoded)
  loss_pt = 0
  loss_g = energy_g + loss_pt

  disc_opt = get_opt(loss_d, "discriminator")
  gen_opt = get_opt(loss_g, "generator")

  decoded_imgs = convert_img(decoded)
  contrastive_samples_imgs = convert_img(contrastive_samples)
  contrastive_decoded_imgs = convert_img(contrastive_decoded)

  start = datetime.now()
  print "Start: ",  start.strftime("%Y-%m-%d_%H-%M-%S")

  num_threads = FLAGS.num_threads
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=num_threads)) as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

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
        if FLAGS.noise_type == 'normal':
          z_samples = np.random.normal(0.0, 1.0, size=[FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)
        else:
          z_samples = np.random.uniform(-1.0, 1.0, size=[FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)
        feed_dict = {z_vecs:z_samples, }
        #print feed_dict[z_vecs][1][:10]

        #sample_val, energy_d_val, loss_d_val, disc_opt_val, decoded_img_val, contrastive_sample_val = sess.run([samples[0], energy_d, loss_d, disc_opt, decoded_imgs[0], contrastive_samples[0]], feed_dict=feed_dict)
        #sess.run([disc_opt], feed_dict=feed_dict)

        #print "WGs0: [", np.sum(WGs[0].eval()), np.sum(WGs[1].eval()), np.sum(WGs[2].eval()), np.sum(WGs[3].eval()), "] WDs0:", np.sum(WDs[0].eval())

        _, _, energy_d_val = sess.run([disc_opt,disc_opt, energy_d], feed_dict=feed_dict)

        #print "WGs1: [", np.sum(WGs[0].eval()), np.sum(WGs[1].eval()), np.sum(WGs[2].eval()), np.sum(WGs[3].eval()), "] WDs0:", np.sum(WDs[0].eval())
        print "------------------------------------------------------"
        energy_g_val, loss_d_val = sess.run([energy_g, loss_d], feed_dict=feed_dict)
        #loss_g_val, loss_pt_val = sess.run([loss_g, loss_pt], feed_dict=feed_dict)
        #print "[%05d]" % itr, "Ed=", energy_d_val, "Eg=", energy_g_val,"Ld(x, z)=", loss_d_val, "Lg(z)=", loss_g_val, "PT=", loss_pt_val
        loss_g_val = sess.run([loss_g], feed_dict=feed_dict)
        print "[%05d]" % itr, "Ed=", energy_d_val, "Eg=", energy_g_val,"Ld(x, z)=", loss_d_val, "Lg(z)=", loss_g_val

        sess.run([gen_opt], feed_dict=feed_dict)
        #sess.run([gen_opt], feed_dict=feed_dict)

        print "------------------------------------------------------"
        energy_d_val, energy_g_val, loss_d_val = sess.run([energy_d, energy_g, loss_d], feed_dict=feed_dict)
        #loss_g_val, loss_pt_val = sess.run([loss_g, loss_pt], feed_dict=feed_dict)
        #print "[%05d]" % itr, "Ed=", energy_d_val, "Eg=", energy_g_val,"Ld(x, z)=", loss_d_val, "Lg(z)=", loss_g_val, "PT=", loss_pt_val
        loss_g_val = sess.run([loss_g], feed_dict=feed_dict)
        print "[%05d]" % itr, "Ed=", energy_d_val, "Eg=", energy_g_val,"Ld(x, z)=", loss_d_val, "Lg(z)=", loss_g_val

        #print "WGs2: [", np.sum(WGs[0].eval()), np.sum(WGs[1].eval()), np.sum(WGs[2].eval()), np.sum(WGs[3].eval()), "] WDs0:", np.sum(WDs[0].eval())


        sample_val, decoded_img_val, contrastive_sample_val, contrastive_decoded_val = sess.run([samples[0], decoded_imgs[0], contrastive_samples_imgs[0], contrastive_decoded_imgs[0]], feed_dict=feed_dict)

        current = datetime.now()
        print "\telapsed:", current - start

        cv2.imshow('decoded', cv2.cvtColor(img_listup(sample_val, decoded_img_val),cv2.COLOR_RGB2BGR)) 
        cv2.imshow('contrastive', cv2.cvtColor(img_listup(contrastive_sample_val, contrastive_decoded_val), cv2.COLOR_RGB2BGR)) 
        cv2.waitKey(5)
        import scipy.misc
        #scipy.misc.imsave("generated"+current.strftime("%Y%m%d_%H%M%S")+".png", contrastive_sample_val)
        scipy.misc.imsave(save_dir + "/generated"+"%02d"%(itr%100)+".png", contrastive_sample_val)
        if itr > 1 and itr % 300 == 0:
          #energy_d_val, loss_d_val, loss_g_val = sess.run([energy_d, loss_d, loss_g])
          print "#######################################################"
          #print "\tE=", energy_d_val, "Ld(x, z)=", loss_d, "Lg(z)=", loss_g
          saver.save(sess, checkpoint)
    except tf.errors.OutOfRangeError:
      print "the last epoch ends."

    coord.request_stop()
    coord.join(threads)

    cv2.destroyAllWindows()


if __name__ == "__main__":
  tf.app.run()
