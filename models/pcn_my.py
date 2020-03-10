import tensorflow as tf
from tf_util import *
import importlib
import tflearn
import numpy as np


class Model:
    def __init__(self, images, inputs, npts, gt, alpha, batch_size, mode=1):
        self.num_coarse = 1024
        self.grid_size = 4
        self.grid_scale = 0.05
        self.batch_size = batch_size
        self.num_fine = self.grid_size ** 2 * self.num_coarse

        if mode == 1:
            # 【pcn点云特征】features_pt  = batch_size * 1024
            self.features_pt = self.create_encoder(inputs, npts)

            # 【psg图像特征】
            with tf.device('/gpu:0'):
                self.features_image = self.create_psg(images)
                self.features_image = mlp(self.features_image, [1024], "mlp_psg")

            # 1.连接2者
            self.features = tf.concat([self.features_pt, self.features_image], axis=1)
            self.features = mlp(self.features, [1024], "mlp_concat")
        elif mode == 2:
            # 【pcn点云特征】features_pt  = batch_size * 1024
            self.features_pt = self.create_encoder(inputs, npts)

            # 2.只有点云输入
            self.features = self.features_pt
        elif mode == 3:
            # 【psg图像特征】
            with tf.device('/gpu:0'):
                self.features_image = self.create_psg(images)
                self.features_image = mlp(self.features_image, [1024], "mlp_psg")
            # 3.只有图像输入
            self.features = self.features_image

        self.coarse, self.fine = self.create_decoder(self.features)
        self.loss, self.update = self.create_loss(self.coarse, self.fine, gt, alpha)
        self.outputs = self.fine
        self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, self.fine, gt]
        self.visualize_titles = ['input', 'coarse output', 'fine output', 'ground truth']

    def create_psg(self, image):
        x = image
        # 192 256
        x = tflearn.layers.conv.conv_2d(x, 16, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 16, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x0 = x
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        # 96 128
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x1 = x
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        # 48 64
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x2 = x
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        # 24 32
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x3 = x
        x = tflearn.layers.conv.conv_2d(x, 256, (5, 5), strides=2, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        # 12 16
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x4 = x
        x = tflearn.layers.conv.conv_2d(x, 512, (5, 5), strides=2, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        # 6 8
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x5 = x
        x = tflearn.layers.conv.conv_2d(x, 512, (5, 5), strides=2, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        # 3 4
        x_additional = tflearn.layers.core.fully_connected(x, 2048, activation='relu', weight_decay=1e-3,
                                                           regularizer='L2')
        x = tflearn.layers.conv.conv_2d_transpose(x, 256, [5, 5], [7, 7], strides=2, activation='linear',
                                                  weight_decay=1e-5, regularizer='L2')
        # 6 8
        x5 = tflearn.layers.conv.conv_2d(x5, 256, (3, 3), strides=1, activation='linear', weight_decay=1e-5,
                                         regularizer='L2')
        x = tf.nn.relu(tf.add(x, x5))
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x5 = x
        x = tflearn.layers.conv.conv_2d_transpose(x, 128, [5, 5], [14, 14], strides=2, activation='linear',
                                                  weight_decay=1e-5, regularizer='L2')
        # 12 16
        x4 = tflearn.layers.conv.conv_2d(x4, 128, (3, 3), strides=1, activation='linear', weight_decay=1e-5,
                                         regularizer='L2')
        x = tf.nn.relu(tf.add(x, x4))
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x4 = x
        x = tflearn.layers.conv.conv_2d_transpose(x, 64, [5, 5], [28, 28], strides=2, activation='linear',
                                                  weight_decay=1e-5, regularizer='L2')
        # 24 32
        x3 = tflearn.layers.conv.conv_2d(x3, 64, (3, 3), strides=1, activation='linear', weight_decay=1e-5,
                                         regularizer='L2')
        x = tf.nn.relu(tf.add(x, x3))
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x3 = x
        x = tflearn.layers.conv.conv_2d_transpose(x, 32, [5, 5], [56, 56], strides=2, activation='linear',
                                                  weight_decay=1e-5, regularizer='L2')
        # 48 64
        x2 = tflearn.layers.conv.conv_2d(x2, 32, (3, 3), strides=1, activation='linear', weight_decay=1e-5,
                                         regularizer='L2')
        x = tf.nn.relu(tf.add(x, x2))
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x2 = x
        x = tflearn.layers.conv.conv_2d_transpose(x, 16, [5, 5], [112, 112], strides=2, activation='linear',
                                                  weight_decay=1e-5, regularizer='L2')
        # 96 128
        x1 = tflearn.layers.conv.conv_2d(x1, 16, (3, 3), strides=1, activation='linear', weight_decay=1e-5,
                                         regularizer='L2')
        x = tf.nn.relu(tf.add(x, x1))
        x = tflearn.layers.conv.conv_2d(x, 16, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=2, activation='linear', weight_decay=1e-5,
                                        regularizer='L2')
        # 48 64
        x2 = tflearn.layers.conv.conv_2d(x2, 32, (3, 3), strides=1, activation='linear', weight_decay=1e-5,
                                         regularizer='L2')
        x = tf.nn.relu(tf.add(x, x2))
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x2 = x
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=2, activation='linear', weight_decay=1e-5,
                                        regularizer='L2')
        # 24 32
        x3 = tflearn.layers.conv.conv_2d(x3, 64, (3, 3), strides=1, activation='linear', weight_decay=1e-5,
                                         regularizer='L2')
        x = tf.nn.relu(tf.add(x, x3))
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x3 = x
        x = tflearn.layers.conv.conv_2d(x, 128, (5, 5), strides=2, activation='linear', weight_decay=1e-5,
                                        regularizer='L2')
        # 12 16
        x4 = tflearn.layers.conv.conv_2d(x4, 128, (3, 3), strides=1, activation='linear', weight_decay=1e-5,
                                         regularizer='L2')
        x = tf.nn.relu(tf.add(x, x4))
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x4 = x
        x = tflearn.layers.conv.conv_2d(x, 256, (5, 5), strides=2, activation='linear', weight_decay=1e-5,
                                        regularizer='L2')
        # 6 8
        x5 = tflearn.layers.conv.conv_2d(x5, 256, (3, 3), strides=1, activation='linear', weight_decay=1e-5,
                                         regularizer='L2')
        x = tf.nn.relu(tf.add(x, x5))
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x5 = x
        x = tflearn.layers.conv.conv_2d(x, 512, (5, 5), strides=2, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        # 3 4
        x_additional = tflearn.layers.core.fully_connected(x_additional, 2048, activation='linear', weight_decay=1e-4,
                                                           regularizer='L2')
        x_additional = tf.nn.relu(tf.add(x_additional, tflearn.layers.core.fully_connected(x, 2048, activation='linear',
                                                                                           weight_decay=1e-3,
                                                                                           regularizer='L2')))
        x = tflearn.layers.conv.conv_2d_transpose(x, 256, [5, 5], [7, 7], strides=2, activation='linear',
                                                  weight_decay=1e-5, regularizer='L2')
        # 6 8
        x5 = tflearn.layers.conv.conv_2d(x5, 256, (3, 3), strides=1, activation='linear', weight_decay=1e-5,
                                         regularizer='L2')
        x = tf.nn.relu(tf.add(x, x5))
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x5 = x
        x = tflearn.layers.conv.conv_2d_transpose(x, 128, [5, 5], [14, 14], strides=2, activation='linear',
                                                  weight_decay=1e-5, regularizer='L2')
        # 12 16
        x4 = tflearn.layers.conv.conv_2d(x4, 128, (3, 3), strides=1, activation='linear', weight_decay=1e-5,
                                         regularizer='L2')
        x = tf.nn.relu(tf.add(x, x4))
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x4 = x
        x = tflearn.layers.conv.conv_2d_transpose(x, 64, [5, 5], [28, 28], strides=2, activation='linear',
                                                  weight_decay=1e-5, regularizer='L2')
        # 24 32
        x3 = tflearn.layers.conv.conv_2d(x3, 64, (3, 3), strides=1, activation='linear', weight_decay=1e-5,
                                         regularizer='L2')
        x = tf.nn.relu(tf.add(x, x3))
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')

        x_additional = tflearn.layers.core.fully_connected(x_additional, 1024, activation='relu', weight_decay=1e-3,
                                                           regularizer='L2')
        x_additional = tflearn.layers.core.fully_connected(x_additional, 240 * 3, activation='linear',
                                                           weight_decay=1e-3, regularizer='L2')
        # x_additional = tf.reshape(x_additional, (self.batch_size, 240, 3))
        # x = tflearn.layers.conv.conv_2d(x, 3, (3, 3), strides=1, activation='linear', weight_decay=1e-5,
        #                                 regularizer='L2')
        # x = tf.reshape(x, (self.batch_size, 28 * 28, 3))
        # #[batch,1024*3]
        # x = tf.concat([x_additional, x], 1)
        # x = tf.reshape(x, (self.batch_size, 1024, 3))
        x = tflearn.layers.conv.conv_2d(x, 3, (3, 3), strides=1, activation='linear', weight_decay=1e-5,
                                        regularizer='L2')

        x = tf.reshape(x, (self.batch_size, 28 * 28 * 3))
        x = tf.concat([x_additional, x], 1)

        return x

    def create_encoder(self, inputs, npts):
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [128, 256])
            features_global = point_unpool(point_maxpool(features, npts, keepdims=True), npts)
            features = tf.concat([features, features_global], axis=2)
        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [512, 1024])
            features = point_maxpool(features, npts)
        return features

    def create_decoder(self, features):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            coarse = mlp(features, [1024, 1024, self.num_coarse * 3], "mlp_decoder")
            coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])

        with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            x = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
            y = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
            grid = tf.meshgrid(x, y)
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            grid_feat = tf.tile(grid, [features.shape[0], self.num_coarse, 1])

            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3])

            global_feat = tf.tile(tf.expand_dims(features, 1), [1, self.num_fine, 1])

            feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            center = tf.reshape(center, [-1, self.num_fine, 3])

            fine = mlp_conv(feat, [512, 512, 3]) + center
        return coarse, fine

    def create_loss(self, coarse, fine, gt, alpha):
        #cd或者emd loss
        # gt_ds = gt[:, :coarse.shape[1], :]
        # loss_coarse = earth_mover(coarse, gt_ds)
        loss_coarse = chamfer(coarse, gt)

        add_train_summary('train/coarse_loss', loss_coarse)
        update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)

        loss_fine = chamfer(fine, gt)
        add_train_summary('train/fine_loss', loss_fine)
        update_fine = add_valid_summary('valid/fine_loss', loss_fine)

        loss = loss_coarse + alpha * loss_fine
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, [update_coarse, update_fine, update_loss]
