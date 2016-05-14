import ipdb
import os
from glob import glob
import pandas as pd
import numpy as np
import cv2
from model import *
from util import *

n_epochs = 10000
learning_rate_val = 0.001
weight_decay_rate = 0.00001
momentum = 0.9
batch_size = 400
lambda_recon = 0.999
lambda_adv = 0.001

overlap_size = 7
hiding_size = 64

testset_path  = '../data/lsun_testset.pickle'
result_path= '../results/lsun/'
pretrained_model_path = '../models/lsun/model-1'
testset = pd.read_pickle( testset_path )

is_train = tf.placeholder( tf.bool )
images_tf = tf.placeholder( tf.float32, [batch_size, 128, 128, 3], name="images")

model = Model()

reconstruction = model.build_reconstruction(images_tf, is_train)

# Applying bigger loss for overlapping region
#sess = tf.InteractiveSession()
#
#tf.initialize_all_variables().run()
#saver.restore( sess, pretrained_model_path )

ii = 0
for start,end in zip(
        range(0, len(testset), batch_size),
        range(batch_size, len(testset), batch_size)):

    test_image_paths = testset[:batch_size]['image_path'].values
    test_images_ori = map(lambda x: load_image(x), test_image_paths)

    test_images_crop = map(lambda x: crop_random(x, x=32, y=32), test_images_ori)
    test_images, test_crops, xs,ys = zip(*test_images_crop)

#    reconstruction_vals = sess.run(
#            reconstruction,
#            feed_dict={
#                images_tf: test_images,
#                images_hiding: test_crops,
#                is_train: False
#                })
    for img,x,y in zip(test_images, xs, ys):
#        rec_hid = (255. * (rec_val+1)/2.).astype(int)
#        rec_con = (255. * (img+1)/2.).astype(int)
#
#        rec_con[y:y+64, x:x+64] = rec_hid
        img_rgb = (255. * (img + 1)/2.).astype(int)
        cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'.ori.jpg'), img_rgb)
        #cv2.imwrite( os.path.join(result_path, 'img_ori'+str(ii)+'.'+str(int(iters/1000))+'.jpg'), rec_con)
        ii += 1
        if ii > 30: break

