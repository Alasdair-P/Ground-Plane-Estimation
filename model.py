import tensorflow as tf


def dilation_model_pretrained(dataset, input_tensor, w_pretrained, trainable):

    def conv(name, input, strides, padding, add_bias, apply_relu, atrous_rate=None):
        """
        Helper function for loading convolution weights from weight dictionary.
        """
        with tf.variable_scope(name):

            # Load kernel weights and apply convolution
            w_kernel = w_pretrained[name + '/kernel:0']
            w_kernel = tf.Variable(initial_value=w_kernel, trainable=trainable)

            if not atrous_rate:
                conv_out = tf.nn.conv2d(input, w_kernel, strides, padding)
            else:
                conv_out = tf.nn.atrous_conv2d(input, w_kernel, atrous_rate, padding)
            if add_bias:
                # Load bias values and add them to conv output
                w_bias = w_pretrained[name + '/bias:0']
                w_bias = tf.Variable(initial_value=w_bias, trainable=trainable)
                conv_out = tf.nn.bias_add(conv_out, w_bias)

            if apply_relu:
                # Apply ReLu nonlinearity
                conv_out = tf.nn.relu(conv_out)

        return conv_out

    # Sanity check on dataset name
    if dataset not in ['cityscapes', 'camvid']:
        raise ValueError('Dataset "{}" not supported.'.format(dataset))

    # Start building the model
    else:

        h = conv('conv1_1', input_tensor, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
        h = conv('conv1_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
        h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1')

        h = conv('conv2_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
        h = conv('conv2_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
        h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2')

        h = conv('conv3_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
        h = conv('conv3_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
        h = conv('conv3_3', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
        h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool3')

        h = conv('conv4_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
        h = conv('conv4_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
        h = conv('conv4_3', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)

        h = conv('conv5_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=2)
        h = conv('conv5_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=2)
        h = conv('conv5_3', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=2)
        h = conv('fc6', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=4)

        h = tf.layers.dropout(h, rate=0.5, name='drop6')
        h = conv('fc7', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
        h = tf.layers.dropout(h, rate=0.5, name='drop7')
        h = conv('final', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad1_1')
        h = conv('ctx_conv1_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad1_2')
        h = conv('ctx_conv1_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)

        h = tf.pad(h, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT', name='ctx_pad2_1')
        h = conv('ctx_conv2_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=2)

        h = tf.pad(h, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='CONSTANT', name='ctx_pad3_1')
        h = conv('ctx_conv3_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=4)

        h = tf.pad(h, [[0, 0], [8, 8], [8, 8], [0, 0]], mode='CONSTANT', name='ctx_pad4_1')
        h = conv('ctx_conv4_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=8)

        h = tf.pad(h, [[0, 0], [16, 16], [16, 16], [0, 0]], mode='CONSTANT', name='ctx_pad5_1')
        h = conv('ctx_conv5_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=16)

        if dataset == 'cityscapes':
            h = tf.pad(h, [[0, 0], [32, 32], [32, 32], [0, 0]], mode='CONSTANT', name='ctx_pad6_1')
            h = conv('ctx_conv6_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=32)

            h = tf.pad(h, [[0, 0], [64, 64], [64, 64], [0, 0]], mode='CONSTANT', name='ctx_pad7_1')
            h = conv('ctx_conv7_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=64)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad_fc1')
        h = conv('ctx_fc1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
        h = conv('ctx_final', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=False)

        if dataset == 'cityscapes':
            h = tf.image.resize_bilinear(h, size=(1024, 1024))
            logits = conv('ctx_upsample', h, strides=[1, 1, 1, 1], padding='SAME', add_bias=False, apply_relu=True)
        else:
            logits = h

        softmax = tf.nn.softmax(logits, dim=3, name='softmax')

    return softmax
