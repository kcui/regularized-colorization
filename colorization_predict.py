import os, sys
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import network, img_io, color_transform

eps = 1e-5

def print_(str, color='', bold=False):
    if color == 'w':
        sys.stdout.write('\033[93m')
    elif color == "e":
        sys.stdout.write('\033[91m')
    elif color == "m":
        sys.stdout.write('\033[95m')

    if bold:
        sys.stdout.write('\033[1m')

    sys.stdout.write(str)
    sys.stdout.write('\033[0m')
    sys.stdout.flush()


# Settings, using TensorFlow arguments
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "10", "Batch size")
tf.flags.DEFINE_integer("width", "1024", "Reconstruction image width")
tf.flags.DEFINE_integer("height", "768", "Reconstruction image height")
tf.flags.DEFINE_string("im_dir", "data", "Path to image directory or an individual image")
tf.flags.DEFINE_string("out_dir", "out", "Path to output directory")
tf.flags.DEFINE_string("params", "hdrcnn_params_compr_regularized.npz", "Path to trained CNN weights")
tf.flags.DEFINE_list("params_ckpt", "-1", "Param checkpoints")
tf.flags.DEFINE_float("scaling", "1.0", "Pre-scaling, which is followed by clipping, in order to remove compression artifacts close to highlights")
tf.flags.DEFINE_float("gamma", "1.0", "Gamma/exponential curve applied before, and inverted after, prediction. This can be used to control the boost of reconstructed pixels.")
tf.flags.DEFINE_bool("vgg", False, "Use VGG encoder")
tf.flags.DEFINE_bool("input", False, "")

# Round to be multiple of 32, so that autoencoder pooling+upsampling
# yields same size as input image
sx = int(np.maximum(32, np.round(FLAGS.width/32.0)*32))
sy = int(np.maximum(32, np.round(FLAGS.height/32.0)*32))
if sx != FLAGS.width or sy != FLAGS.height:
    print_("Warning: ", 'w', True)
    print_("prediction size has been changed from %dx%d pixels to %dx%d\n"%(FLAGS.width, FLAGS.height, sx, sy), 'w')
    print_("         pixels, to comply with autoencoder pooling and up-sampling.\n\n", 'w')

# Info
print_("\n\n\t-------------------------------------------------------------------\n", 'm')
print_("\t  Colorization\n\n", 'm')
print_("\t  Prediction settings\n", 'm')
print_("\t  -------------------\n", 'm')
print_("\t  Input image directory/file:     %s\n" % FLAGS.im_dir, 'm')
print_("\t  Output directory:               %s\n" % FLAGS.out_dir, 'm')
print_("\t  CNN weights:                    %s\n" % FLAGS.params, 'm')
print_("\t  Prediction resolution:          %dx%d pixels\n" % (sx, sy), 'm')
if FLAGS.scaling > 1.0:
    print_("\t  Pre-scaling:                    %0.4f\n" % FLAGS.scaling, 'm')
if FLAGS.gamma > 1.0 + eps or FLAGS.gamma < 1.0 - eps:
    print_("\t  Gamma:                          %0.4f\n" % FLAGS.gamma, 'm')
print_("\t-------------------------------------------------------------------\n\n\n", 'm')

# Single frame
frames = [FLAGS.im_dir]

# If directory is supplied, get names of all files in the path
if os.path.isdir(FLAGS.im_dir):
    frames = [os.path.join(FLAGS.im_dir, name)
              for name in sorted(os.listdir(FLAGS.im_dir))
              if os.path.isfile(os.path.join(FLAGS.im_dir, name))]

# Placeholder for image input
y_ = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, sy, sx, 3])

y_lab = color_transform.rgb_to_lab(y_)
y_l, y_a, y_b = color_transform.preprocess_lab(y_lab)
y_a = tf.reshape(y_a, [FLAGS.batch_size, sy, sx, 1])
y_b = tf.reshape(y_b, [FLAGS.batch_size, sy, sx, 1])
y_a = tf.image.resize_images(y_a, size=[int(sx/2),int(sy/2)], method=tf.image.ResizeMethod.BILINEAR)
y_b = tf.image.resize_images(y_b, size=[int(sx/2),int(sy/2)], method=tf.image.ResizeMethod.BILINEAR)
x = tf.reshape(y_l, [FLAGS.batch_size, sy, sx, 1])

# HDR reconstruction autoencoder model
print_("Network setup:\n")

if FLAGS.vgg:
  print_('\nEncoder model from VGG16.\n')
  x = tf.stack([x,x,x], axis=3)
  x = tf.reshape(x, [FLAGS.batch_size,sy,sx,3])
  with tf.variable_scope("siamese") as scope:
    yab = network.model_vgg(x, False, True)
else:
  print_('\nEncoder model ala Izuka et al.\n')
  with tf.variable_scope("siamese") as scope:
    yab = network.model(x, False, True)
    scope.reuse_variables()
    yab_s = network.model(x, False, True)
  #yab = network.model(x, False, True)

# Compose final image
ya, yb = tf.unstack(yab, axis=3)
ya = tf.reshape(ya, [FLAGS.batch_size, int(sy/2), int(sx/2), 1])
yb = tf.reshape(yb, [FLAGS.batch_size, int(sy/2), int(sx/2), 1])
yabf = tf.image.resize_images(yab, size=[sy,sx], method=tf.image.ResizeMethod.BILINEAR)
yaf, ybf = tf.unstack(yabf, axis=3)
ylab = color_transform.deprocess_lab(y_l, yaf, ybf)
y = color_transform.lab_to_rgb(ylab)

saver = tf.train.Saver()

# TensorFlow session for running inference
#sess = tf.InteractiveSession()
with tf.Session() as sess:

    for c in range(len(FLAGS.params_ckpt)):
        if int(FLAGS.params_ckpt[c]) < 0:
            params = FLAGS.params
        else:
            params = FLAGS.params%int(FLAGS.params_ckpt[c])

        # Load trained CNN weights
        print_("\nLoading trained parameters from '%s'..."%params)
        saver.restore(sess, params)
        print_("\tdone\n")

        if int(FLAGS.params_ckpt[c]) < 0:
            out_dir = FLAGS.out_dir
        else:
            out_dir = FLAGS.out_dir%int(FLAGS.params_ckpt[c])

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        print_("\nStarting prediction...\n\n")
        k, j = 0, 0
        y_buffer = np.empty([FLAGS.batch_size,sy,sx,3], dtype=np.float32)
        for i in range(len(frames)):
            print("Frame %d: '%s'"%(i,frames[i]))

            try:
                # Read frame
                print_("\tReading...")
                y_read = img_io.readLDR(frames[i], (sy,sx), True, FLAGS.scaling)
                print_("\tdone")

                y_buffer[j,:,:,:] = y_read
                j += 1

                if j == FLAGS.batch_size:
                    j = 0

                    # Run prediction.
                    # The gamma value is used to allow for boosting/reducing the intensity of
                    # the reconstructed highlights. If y = f(x) is the reconstruction, the gamma
                    # g alters this according to y = f(x^(1/g))^g
                    print_("\tInference...")
                    feed_dict = {y_: np.power(np.maximum(y_buffer, 0.0), 1.0/FLAGS.gamma)}
                    if FLAGS.input:
                        x_buffer = sess.run(x, feed_dict=feed_dict)
                    else:
                        x_buffer, y_predict = sess.run([x,y], feed_dict=feed_dict)
                    print_("\tdone\n")

                    # Write to disc
                    print_("\tWriting...")
                    for b in range(FLAGS.batch_size):
                        k += 1;
                        #img_io.writeLDR(np.squeeze(y_buffer[b,:,:,:]), '%s/%06d_gt.png' % (out_dir, k), -3)
                        if FLAGS.input:
                            img_io.writeLDR(np.squeeze(x_buffer[b,:,:]), '%s/%06d.png' % (out_dir, k), -3)
                        else:
                            img_io.writeLDR(np.squeeze(y_predict[b,:,:,:]), '%s/%06d.png' % (out_dir, k), -3)
                    print_("\tdone\n")

            except img_io.IOException as e:
                print_("\n\t\tWarning! ", 'w', True)
                print_("%s\n"%e, 'w')
            except Exception as e:    
                print_("\n\t\tError: ", 'e', True)
                print_("%s\n"%e, 'e')

        print_("Done!\n")


