'''
TAKEN FROM: https://github.com/ndrplz/dilation-tensorflow
'''

import tensorflow as tf
import pickle
import cv2
import os
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
from utils import predict
from model import dilation_model_pretrained
from datasets import CONFIG

def main():
    # Choose between 'cityscapes' and 'camvid'
    dataset = 'cityscapes'

    # Load dict of pretrained weights
    print('Loading pre-trained weights...')
    with open(CONFIG[dataset]['weights_file'], 'rb') as f:
        w_pretrained = pickle.load(f)
    print('Done.')

    # Create checkpoint directory
    checkpoint_dir = path.join('data/checkpoint', 'dilation_' + dataset)
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    Path = '/Users/alasdair/Desktop/Thesis_git/Data/test_ims'
    Path_out = '/Users/alasdair/Desktop/Thesis_git/Data/outputs'
    im_names = []
    for filename in os.listdir(Path):
        im_names.append(filename)
    im_names.sort()
    print('Number of images: ', len(im_names))
    masks = np.zeros((256, 512, len(im_names)))

    # Build pretrained model and save it as TF checkpoint
    with tf.Session() as sess:

        # Choose input shape according to dataset characteristics
        input_h, input_w, input_c = CONFIG[dataset]['input_shape']
        input_tensor = tf.placeholder(tf.float32, shape=(None, input_h, input_w, input_c), name='input_placeholder')

        # Create pretrained model
        model = dilation_model_pretrained(dataset, input_tensor, w_pretrained, trainable=False)

        sess.run(tf.global_variables_initializer())

        # Save both graph and weights
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        saver.save(sess, path.join(checkpoint_dir, 'dilation'))

    # Restore both graph and weights from TF checkpoint
    with tf.Session() as sess:

        saver = tf.train.import_meta_graph(path.join(checkpoint_dir, 'dilation.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        graph = tf.get_default_graph()
        model = graph.get_tensor_by_name('softmax:0')
        model = tf.reshape(model, shape=(1,) + CONFIG[dataset]['output_shape'])

        for i in range(len(im_names)):
            file_n = im_names[i]
            input_image_path = os.path.join(Path, file_n)
            input_image = cv2.resize(cv2.imread(input_image_path), (2048, 1024))
            print('Reading: ', file_n)

            output_image_path = path.join(Path_out, file_n[:-4] + '_out.png')
            output_mask_path = path.join(Path_out, file_n[:-4] + '_mask.png')

            # Read and predict on a test image
            # input_image = cv2.imread(input_image_path)
            input_tensor = graph.get_tensor_by_name('input_placeholder:0')
            predicted_image = predict(input_image, input_tensor, model, dataset, sess)

            # Convert colorspace (palette is in RGB) and save prediction result
            predicted_image = np.array(cv2.resize(predicted_image, (512, 256)))
            mask = np.sum((predicted_image >= 250).astype(int), axis=2)
            masks[:, :, i] = mask

            predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
            print(output_image_path, i)
            cv2.imwrite(output_image_path, predicted_image)
            np.save(os.path.join(Path + 'masks.npy'), mask)


if __name__ == '__main__':
    main()


            





