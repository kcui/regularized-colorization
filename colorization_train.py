import time, math, os, sys, random
import tensorflow as tf
import tensorlayer as tl
import threading
import numpy as np
import scipy.stats as st
import scipy.misc

import network, img_io, color_transform


#=== Settings =================================================================

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("sx",               "128",   "Image width")
tf.flags.DEFINE_integer("sy",               "128",   "Image height")
tf.flags.DEFINE_integer("num_threads",      "4",     "Number of threads for multi-threaded loading of data")
tf.flags.DEFINE_integer("print_batch_freq", "5000",  "Frequency for printing stats and saving images/parameters")
tf.flags.DEFINE_integer("print_batches",    "5",     "Number of batches to output images for at each [print_batch_freq] step")
tf.flags.DEFINE_bool("print_im",            "true",  "If LDR sample images should be printed at each [print_batch_freq] step")

# Paths
tf.flags.DEFINE_string("data_dir",          "", "Path to processed dataset.")
tf.flags.DEFINE_string("output_dir",        "training_output", "Path to output directory, for weights and intermediate results")
tf.flags.DEFINE_string("vgg_path",          "vgg16_places365_weights.npy", "Path to VGG16 pre-trained weigths, for encoder convolution layers")
tf.flags.DEFINE_string("parameters",        "model_trained.npz", "Path to trained params for complete network")
tf.flags.DEFINE_bool("load_params",         "false", "Load the parameters from the [parameters] path, otherwise the parameters from [vgg_path] will be used")
tf.flags.DEFINE_bool("vgg",                 "false", "VGG16 encoder net")

# Learning parameters
tf.flags.DEFINE_float("num_epochs",         "10.0",   "Number of training epochs")
tf.flags.DEFINE_float("start_step",         "0.0",     "Step to start from")
tf.flags.DEFINE_float("learning_rate",      "0.0001",  "Starting learning rate for Adam optimizer")
tf.flags.DEFINE_integer("batch_size",       "4",       "Batch size for training")
tf.flags.DEFINE_bool("rand_data",           "true",    "Random shuffling of training data")
tf.flags.DEFINE_float("train_size",         "0.99",    "Fraction of data to use for training, the rest is validation data")
tf.flags.DEFINE_integer("buffer_size",      "256",     "Size of load queue when reading training data")

# Regularization for temporal stability
tf.flags.DEFINE_integer("regularization",     "0",     "Use regularization: 0 = none, 1 = coherence, 2 = sparse jacobian")
tf.flags.DEFINE_float("regularization_alpha", "0.95",  "Regularization strength")
tf.flags.DEFINE_float("transf_scaling",       "1.0",   "Magnitude of transformations")
tf.flags.DEFINE_bool("noise",                 "false", "Apply noise to transformed images")

#==============================================================================

sx =  FLAGS.sx
sy =  FLAGS.sy
data_dir = FLAGS.data_dir
log_dir = os.path.join(FLAGS.output_dir, "logs")
im_dir = os.path.join(FLAGS.output_dir, "im")


#=== Localize training data ===================================================

# Get names of all images in the training path

frames = []
# r=root, d=directories, f = files
for r, d, f in os.walk("data\\images\\train"):
  for file in f:
    if '.jpg' in file:
      frames.append(os.path.join(r, file))

#frames = [name for name in sorted(os.listdir(data_dir)) if os.path.isfile(os.path.join(data_dir, name))]

# Randomize the images
if FLAGS.rand_data:
  random.shuffle(frames)

# Split data into training/validation sets
splitPos = len(frames) - math.floor(max(FLAGS.batch_size, min((1-FLAGS.train_size)*len(frames), 1000)))
frames_train, frames_valid = np.split(frames, [splitPos])

# Number of steps per epoch depends on the number of training images
training_samples = len(frames_train)
validation_samples = len(frames_valid)
steps_per_epoch = training_samples/FLAGS.batch_size

print("\n\nData to be used:")
print("\t%d training images" % training_samples)
print("\t%d validation images\n" % validation_samples)


#=== Load validation data =====================================================

# Load all validation images into memory
print("Loading validation data...")
y_valid = []
for i in range(len(frames_valid)):
  if i % 10 == 0:
    print("\tframe %d of %d" % (i, len(frames_valid)))
    sys.stdout.flush()
    
  yv = scipy.misc.imread(os.path.join(data_dir, frames_valid[i])).astype(np.float32)/255.0
  yv = yv[np.newaxis,:,:,:]

  if i == 0:
    y_valid = yv
  else:
    y_valid = np.concatenate((y_valid, yv), axis=0)
print("...done!\n\n")
sys.stdout.flush()

del frames


#=== Setup data queues ========================================================

# For single-threaded queueing of frame names
input_frame = tf.placeholder(tf.string)
q_frames = tf.FIFOQueue(FLAGS.buffer_size, [tf.string])
enqueue_op_frames = q_frames.enqueue([input_frame])
dequeue_op_frames = q_frames.dequeue()

# For multi-threaded queueing of training images
input_data = tf.placeholder(tf.float32, shape=[sy, sx, 3])
q_train = tf.FIFOQueue(FLAGS.buffer_size, [tf.float32], shapes=[sy,sx,3])
enqueue_op_train = q_train.enqueue([input_data])
y_ = q_train.dequeue_many(FLAGS.batch_size)


#=== Random transformation for regularization =================================
y_aug = y_

if FLAGS.regularization > 0:
  sc = FLAGS.transf_scaling

  # Random transformation of translation, rotation, zoom, and shearing
  tx = tf.random_uniform(shape=[FLAGS.batch_size,1], minval=-2.0*sc, maxval=2.0*sc, dtype=tf.float32)
  ty = tf.random_uniform(shape=[FLAGS.batch_size,1], minval=-2.0*sc, maxval=2.0*sc, dtype=tf.float32)
  r  = tf.random_uniform(shape=[FLAGS.batch_size,1], minval=np.deg2rad(-sc), maxval=np.deg2rad(sc), dtype=tf.float32)
  z  = tf.random_uniform(shape=[FLAGS.batch_size,1], minval=1.0-0.03*sc, maxval=1.0+0.03*sc, dtype=tf.float32)
  hx = tf.random_uniform(shape=[FLAGS.batch_size,1], minval=np.deg2rad(-sc), maxval=np.deg2rad(sc), dtype=tf.float32)
  hy = tf.random_uniform(shape=[FLAGS.batch_size,1], minval=np.deg2rad(-sc), maxval=np.deg2rad(sc), dtype=tf.float32)
  a = hx - r
  b = tf.cos(hx)
  c = hy + r
  d = tf.cos(hy)
  m1 = tf.divide(z*tf.cos(a), b)
  m2 = tf.divide(z*tf.sin(a), b)
  m3 = tf.divide(sx*b-sx*z*tf.cos(a)+2*tx*z*tf.cos(a)-sy*z*tf.sin(a)+2*ty*z*tf.sin(a), 2*b)
  m4 = tf.divide(z*tf.sin(c), d)
  m5 = tf.divide(z*tf.cos(c), d)
  m6 = tf.divide(sy*d-sy*z*tf.cos(c)+2*ty*z*tf.cos(c)-sx*z*tf.sin(c)+2*tx*z*tf.sin(c), 2*d)
  m7 = tf.zeros([FLAGS.batch_size,2], 'float32')
  transf = tf.concat([m1, m2, m3, m4, m5, m6, m7], 1)
  y_aug = tf.contrib.image.transform(y_aug, transf, interpolation='BILINEAR')

if FLAGS.noise:
  std = tf.random_uniform(shape=[1], minval=0.01, maxval=0.05, dtype=tf.float32)
  y_aug = tf.add(y_aug, tf.random_normal(shape=tf.shape(y_aug), mean=0.0, stddev=std, dtype=tf.float32))
  y_aug = tf.minimum(1.0, tf.maximum(0.0, y_aug))

#=== Network ==================================================================


#
y_lab = color_transform.rgb_to_lab(y_)
y_l, y_a, y_b = color_transform.preprocess_lab(y_lab)
y_a = tf.reshape(y_a, [FLAGS.batch_size, sx, sy, 1])
y_b = tf.reshape(y_b, [FLAGS.batch_size, sx, sy, 1])
y_a = tf.image.resize_images(y_a, size=[int(sx/2),int(sy/2)], method=tf.image.ResizeMethod.BILINEAR)
y_b = tf.image.resize_images(y_b, size=[int(sx/2),int(sy/2)], method=tf.image.ResizeMethod.BILINEAR)
x = tf.reshape(y_l, [FLAGS.batch_size, sx, sy, 1])

#
y_aug_lab = color_transform.rgb_to_lab(y_aug)
y_aug_l, y_aug_a, y_aug_b = color_transform.preprocess_lab(y_aug_lab)
y_aug_a = tf.reshape(y_aug_a, [FLAGS.batch_size, sx, sy, 1])
y_aug_b = tf.reshape(y_aug_b, [FLAGS.batch_size, sx, sy, 1])
y_aug_a = tf.image.resize_images(y_aug_a, size=[int(sx/2),int(sy/2)], method=tf.image.ResizeMethod.BILINEAR)
y_aug_b = tf.image.resize_images(y_aug_b, size=[int(sx/2),int(sy/2)], method=tf.image.ResizeMethod.BILINEAR)
x_aug = tf.reshape(y_aug_l, [FLAGS.batch_size, sx, sy, 1])

# Model

if FLAGS.vgg:
  print('\nEncoder model from VGG16.\n')
  x = tf.stack([x,x,x], axis=3)
  x = tf.reshape(x, [FLAGS.batch_size,sx,sy,3])
  x_aug = tf.stack([x_aug,x_aug,x_aug], axis=3)
  x_aug = tf.reshape(x_aug, [FLAGS.batch_size,sx,sy,3])
  with tf.variable_scope("siamese") as scope:
    yab, vgg16_conv_layers = network.model_vgg(x, True, True)
    scope.reuse_variables()
    yab_s, vgg16_conv_layers_s = network.model_vgg(x_aug, True, True)
else:
  print('\nEncoder model ala Izuka et al.\n')
  with tf.variable_scope("siamese") as scope:
    yab = network.model(x, True, True)
    scope.reuse_variables()
    yab_s = network.model(x_aug, True, True)

print('Model size = %d weights\n'%network.count_all_vars())

# Compose final image
ya, yb = tf.unstack(yab, axis=3)
ya = tf.reshape(ya, [FLAGS.batch_size, int(sx/2), int(sy/2), 1])
yb = tf.reshape(yb, [FLAGS.batch_size, int(sx/2), int(sy/2), 1])
yabf = tf.image.resize_images(yab, size=[sx,sy], method=tf.image.ResizeMethod.BILINEAR)
yaf, ybf = tf.unstack(yabf, axis=3)
ylab = color_transform.deprocess_lab(y_l, yaf, ybf)
y = color_transform.lab_to_rgb(ylab)

if FLAGS.regularization > 0:
  # Transformed input
  ya_s, yb_s = tf.unstack(yab_s, axis=3)
  ya_s = tf.reshape(ya_s, [FLAGS.batch_size, int(sx/2), int(sy/2), 1])
  yb_s = tf.reshape(yb_s, [FLAGS.batch_size, int(sx/2), int(sy/2), 1])
  ylab_s = color_transform.deprocess_lab(tf.reshape(y_aug_l, [FLAGS.batch_size, sx, sy]), 
                                   tf.reshape(tf.image.resize_images(ya_s, size=[sx,sy], method=tf.image.ResizeMethod.BILINEAR), [FLAGS.batch_size, sx, sy]),
                                   tf.reshape(tf.image.resize_images(yb_s, size=[sx,sy], method=tf.image.ResizeMethod.BILINEAR), [FLAGS.batch_size, sx, sy]))
  y_s = color_transform.lab_to_rgb(ylab_s)

  #Transformed output
  y_tr = tf.contrib.image.transform(tf.stack([y_l, yaf, ybf], axis=3), transf, interpolation='BILINEAR')
  yt_l, yt_a, yt_b = tf.unstack(y_tr, axis=3)
  yt_lab = color_transform.deprocess_lab(yt_l, yt_a, yt_b)
  y_t = color_transform.lab_to_rgb(yt_lab)
  _, yt_a, yt_b = tf.unstack(tf.image.resize_images(y_tr, size=[int(sx/2),int(sy/2)], method=tf.image.ResizeMethod.BILINEAR), axis=3)
  yt_a = tf.reshape(yt_a, [FLAGS.batch_size, int(sx/2), int(sy/2), 1])
  yt_b = tf.reshape(yt_b, [FLAGS.batch_size, int(sx/2), int(sy/2), 1])

# The TensorFlow session to be used
sess = tf.InteractiveSession()


#=== Loss function formulation ================================================

cost = tf.reduce_mean(tf.pow(ya - y_a, 2) + tf.pow(yb - y_b, 2))
cost_input_output = tf.reduce_mean(tf.pow(y_a, 2) + tf.pow(y_b, 2))
print('Using L2 loss\n')

if FLAGS.regularization == 1:
  cost = (1.0-FLAGS.regularization_alpha)*cost + FLAGS.regularization_alpha*tf.reduce_mean(tf.pow(yt_a - ya_s, 2) + tf.pow(yt_b - yb_s, 2))
  print('Using coherence regularization, strength = %f\n'%FLAGS.regularization_alpha)

elif FLAGS.regularization == 2:
  cost = (1.0-FLAGS.regularization_alpha)*cost + FLAGS.regularization_alpha*tf.reduce_mean(tf.pow((ya-ya_s) - (y_a-y_aug_a), 2) + tf.pow((yb-yb_s) - (y_b-y_aug_b), 2))
  print('Using sparse Jacobian regularization, strength = %f\n'%FLAGS.regularization_alpha)

elif FLAGS.regularization == 3:
  cost = (1.0-FLAGS.regularization_alpha)*cost + FLAGS.regularization_alpha*tf.reduce_mean(tf.pow(ya-ya_s, 2) + tf.pow(yb-yb_s, 2))
  print('Using stability regularization, strength = %f\n'%FLAGS.regularization_alpha)

sys.stdout.flush()


# Optimizer
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = FLAGS.learning_rate
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                             int(1e5/FLAGS.batch_size), 0.99, staircase=True)
  train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
          epsilon=1e-8, use_locking=False).minimize(cost, global_step=global_step)


#=== Data enqueueing functions ================================================

# For enqueueing of frame names
def enqueue_frames(enqueue_op, coord, frames):
    
  num_frames = len(frames)
  i, k = 0, 0

  try:
    while not coord.should_stop():
      if k >= training_samples*FLAGS.num_epochs:
        sess.run(q_frames.close())
        break

      if i == num_frames:
        i = 0
        if FLAGS.rand_data:
          random.shuffle(frames)

      fname = frames[i];

      i += 1
      k += 1
      sess.run(enqueue_op, feed_dict={input_frame: fname})
  except tf.errors.OutOfRangeError:
    pass
  except Exception as e:
    coord.request_stop(e)

# For multi-threaded reading and enqueueing of frames
def load_and_enqueue(enqueue_op, coord):
  try:
    while not coord.should_stop():
      fname = sess.run(dequeue_op_frames).decode("utf-8")

      # Load image
      input_data_r = scipy.misc.imread(os.path.join(data_dir, fname)).astype(np.float32)/255.0
      sess.run(enqueue_op, feed_dict={input_data: input_data_r})
  except Exception as e:
    try:
      sess.run(q_train.close())
    except Exception as e:
      pass


#=== Error and output function ================================================

# For calculation of loss and output of intermediate validations images to disc
def calc_loss_and_print(y_data, print_dir, step, N):
  val_loss, orig_loss, n_batch = 0, 0, 0
  for b in range(int(y_data.shape[0]/FLAGS.batch_size)):
    y_batch = y_data[b*FLAGS.batch_size:(b+1)*FLAGS.batch_size,:,:,:]
    feed_dict = {y_: y_batch}

    if FLAGS.regularization > 0:
      err1,err2,xeval,ypred,xtpred,ytpred = sess.run([cost, cost_input_output, y_l, y, y_s, y_t], feed_dict=feed_dict)
    else:
      err1,err2,xeval,ypred = sess.run([cost, cost_input_output, y_l, y], feed_dict=feed_dict)


    val_loss += err1; orig_loss += err2; n_batch += 1
    batch_dir = print_dir

    if y_data.shape[0] > y_batch.shape[0]:
      batch_dir = '%s/batch_%03d' % (print_dir, n_batch)

    if n_batch <= N or N < 0:
      if not os.path.exists(batch_dir):
        os.makedirs(batch_dir)
      for i in range(0, y_batch.shape[0]):
        # Print LDR samples
        if FLAGS.print_im:
          img_io.writeLDR(np.squeeze(xeval[i]),    "%s/%06d_%03d_in.png" % (batch_dir, step, i+1))
          img_io.writeLDR(np.squeeze(y_batch[i]),  "%s/%06d_%03d_gt.png" % (batch_dir, step, i+1))
          img_io.writeLDR(np.squeeze(ypred[i]),    "%s/%06d_%03d_out.png" % (batch_dir, step, i+1))

          if FLAGS.regularization > 0:
            img_io.writeLDR(np.squeeze(xtpred[i]), "%s/%06d_%03d_out_R.png" % (batch_dir, step, i+1))
            img_io.writeLDR(np.squeeze(ytpred[i]), "%s/%06d_%03d_out_T.png" % (batch_dir, step, i+1))

  return (val_loss/n_batch, orig_loss/n_batch)


#=== Setup threads and load parameters ========================================

# Summary for Tensorboard
tf.summary.scalar("learning_rate", learning_rate)
summaries = tf.summary.merge_all()
file_writer = tf.summary.FileWriter(log_dir, sess.graph)

sess.run(tf.global_variables_initializer())

# Threads and thread coordinator
coord = tf.train.Coordinator()
thread1 = threading.Thread(target=enqueue_frames, args=[enqueue_op_frames, coord, frames_train])
thread2 = [threading.Thread(target=load_and_enqueue, args=[enqueue_op_train, coord]) for i in range(FLAGS.num_threads)]
thread1.start()
for t in thread2:
  t.start()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

saver = tf.train.Saver(max_to_keep=1000)

# Loading model weights
if(FLAGS.load_params):
  # Load model weights
  print("\n\nLoading trained parameters from '%s'..." % FLAGS.parameters)
  saver.restore(sess, FLAGS.parameters)
  print("...done!\n")
elif FLAGS.vgg:
    # Load pretrained VGG16 weights for encoder
    print("\n\nLoading parameters for VGG16 convolutional layers, from '%s'..." % FLAGS.vgg_path)
    network.load_vgg_weights(vgg16_conv_layers, FLAGS.vgg_path, sess)
    print("...done!\n")


#=== Run training loop ========================================================

print("\nStarting training...\n")
sys.stdout.flush()

step = FLAGS.start_step
train_loss = 0.0
start_time = time.time()
start_time_tot = time.time()


# The training loop
try:
  while not coord.should_stop():
    step += 1

    _, err_t = sess.run([train_op,cost])

    train_loss += err_t

    # Statistics on intermediate progress
    v = int(max(1.0,FLAGS.print_batch_freq/5.0))
    if (int(step) % v)  == 0:
      val_loss, n_batch = 0, 0

      # Validation loss
      for b in range(int(y_valid.shape[0]/FLAGS.batch_size)):
        y_batch = y_valid[b*FLAGS.batch_size:(b+1)*FLAGS.batch_size,:,:,:]
        feed_dict = {y_: y_batch}
        err = sess.run(cost, feed_dict=feed_dict)
        val_loss += err; n_batch += 1

      # Training and validation loss for Tensorboard
      train_summary = tf.Summary()
      valid_summary = tf.Summary()
      valid_summary.value.add(tag='validation_loss',simple_value=val_loss/n_batch)
      file_writer.add_summary(valid_summary, step)
      train_summary.value.add(tag='training_loss',simple_value=train_loss/v)
      file_writer.add_summary(train_summary, step)

      # Other statistics for Tensorboard
      summary = sess.run(summaries)
      file_writer.add_summary(summary, step)
      file_writer.flush()

      # Intermediate training statistics
      print('  [Step %06d of %06d. Processed %06d of %06d samples. Train loss = %0.6f, valid loss = %0.6f]' % (step, steps_per_epoch*FLAGS.num_epochs, (step % steps_per_epoch)*FLAGS.batch_size, training_samples, train_loss/v, val_loss/n_batch))
      sys.stdout.flush()
      train_loss = 0.0

    # Print statistics, and save weights and some validation images
    if step % FLAGS.print_batch_freq == 0:
      duration = time.time() - start_time
      duration_tot = time.time() - start_time_tot

      print_dir = '%s/step_%06d' % (im_dir, step)
      val_loss, orig_loss = calc_loss_and_print(y_valid, print_dir, step, FLAGS.print_batches)

      # Training statistics
      print('\n')
      print('-------------------------------------------')
      print('Currently at epoch %0.2f of %d.' % (step/steps_per_epoch, FLAGS.num_epochs))
      print('Valid loss input   = %.5f' % (orig_loss))
      print('Valid loss trained = %.5f' % (val_loss))
      print('Timings:')
      print('       Since last: %.3f sec' % (duration))
      print('         Per step: %.3f sec' % (duration/FLAGS.print_batch_freq))
      print('        Per epoch: %.3f sec' % (duration*steps_per_epoch/FLAGS.print_batch_freq))
      print('')
      print('   Per step (avg): %.3f sec' % (duration_tot/step))
      print('  Per epoch (avg): %.3f sec' % (duration_tot*steps_per_epoch/step))
      print('')
      print('       Total time: %.3f sec' % (duration_tot))
      print('   Exp. time left: %.3f sec' % (duration_tot*steps_per_epoch*FLAGS.num_epochs/step - duration_tot))
      print('-------------------------------------------')
      sys.stdout.flush()

      # Save current weights
      save_path = saver.save(sess, '%s/model.ckpt'%log_dir, global_step=int(step))
      print('Model saved in path: %s' % save_path)
      print('\n')

      start_time = time.time()

except tf.errors.OutOfRangeError:
    print('Done!')
except Exception as e:
    print("ERROR: ", e)


#=== Final stats and weights ==================================================

duration = time.time() - start_time
duration_tot = time.time() - start_time_tot

print_dir = '%s/step_%06d' % (im_dir, step)
val_loss, orig_loss = calc_loss_and_print(y_valid, print_dir, step, FLAGS.print_batches)

# Final statistics
print('\n')
print('-------------------------------------------')
print('Finished at epoch %0.2f of %d.' % (step/steps_per_epoch, FLAGS.num_epochs))
print('Valid loss input   = %.5f' % (orig_loss))
print('Valid loss trained = %.5f' % (val_loss))
print('Timings:')
print('   Per step (avg): %.3f sec' % (duration_tot/step))
print('  Per epoch (avg): %.3f sec' % (duration_tot*steps_per_epoch/step))
print('')
print('       Total time: %.3f sec' % (duration_tot))
print('-------------------------------------------')

# Save final weights
save_path = saver.save(sess, '%s/model.ckpt'%log_dir, global_step=int(step))
print('Model saved in path: %s' % save_path)
print('\n')


#=== Shut down ================================================================

# Stop threads
print("Shutting down threads...")
try:
  coord.request_stop()
except Exception as e:
  print("ERROR: ", e)

# Wait for threads to finish
print("Waiting for threads...")
coord.join(threads)

file_writer.close()
sess.close()