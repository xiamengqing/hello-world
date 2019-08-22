import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

# 导入MNSIT数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'/MNIST_data')
'''
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting /MNIST_data\train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting /MNIST_data\train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting /MNIST_data\t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting /MNIST_data\t10k-labels-idx1-ubyte.gz

'''
############################################## 定义模块 ################################################

# 定义输入层（噪声和真实图片）
def get_inputs(noise_dim, image_height, image_width, image_depth):
    """
    :param noise_dim: 噪声图片的size
    :param image_height: 真实图像的height
    :param image_width: 真实图像的width
    :param image_depth: 真实图像的depth
    """
    inputs_real = tf.placeholder(tf.float32, [None, image_height, image_width, image_depth], name='inputs_real')
    inputs_noise = tf.placeholder(tf.float32, [None, noise_dim], name='inputs_noise')

    return inputs_real, inputs_noise


# 定义生成器
# 架构：'输入(100x1)' --> 'FC(4*4*512)' --(reshape)-> 'layer1(4x4x512)' --(leaky_ReLU+dropout+deconv)-> 'layer2(7x7x256)'
#       --(BN+leaky_ReLU+dropout+deconv)-> 'layer3(14x14x128)' --(BN+leaky_ReLU+dropout+dconv)-> 'logits(28x28x1)'
#       --(tanh)-> outputs
def get_generator(noise_img, output_dim, is_train=True, alpha=0.01):
    with tf.variable_scope("generator", reuse=(not is_train)):
        # 100 x 1 to 4 x 4 x 512
        # 全连接层
        layer1 = tf.layers.dense(noise_img, 4 * 4 * 512)
        layer1 = tf.reshape(layer1, [-1, 4, 4, 512])
        # batch normalization
        layer1 = tf.layers.batch_normalization(layer1, training=is_train)
        # Leaky ReLU
        layer1 = tf.maximum(alpha * layer1, layer1)
        # dropout
        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)
        # 4 x 4 x 512 to 7 x 7 x 256
        layer2 = tf.layers.conv2d_transpose(layer1, 256, 4, strides=1, padding='valid')
        layer2 = tf.layers.batch_normalization(layer2, training=is_train)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)
        # 7 x 7 256 to 14 x 14 x 128
        layer3 = tf.layers.conv2d_transpose(layer2, 128, 3, strides=2, padding='same')
        layer3 = tf.layers.batch_normalization(layer3, training=is_train)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)
        # 14 x 14 x 128 to 28 x 28 x 1
        logits = tf.layers.conv2d_transpose(layer3, output_dim, 3, strides=2, padding='same')
        # MNIST原始数据集的像素范围在0-1，这里的生成图片范围为(-1,1)
        # 因此在训练时，记住要把MNIST像素范围进行resize
        outputs = tf.tanh(logits)
        return outputs


# 定义判别器
# 架构：'输入(28x28x1)' --(conv+leaky_ReLU+drop)-> 'layer1(14x14x128)' --(conv+BN+leaky_ReLU+dropout)-> 'layer2(7x7x256)'
#      --(conv+BN+leaky_ReLU+dropout)-> 'layer3(4x4x512)' --(flatten)-> 'flatten(4*4*512)' --(FC)-> 'logits(1)'
#      --(sigmoid)-> 'outputs'
def get_discriminator(inputs_img, reuse=False, alpha=0.01):
    with tf.variable_scope("discriminator", reuse=reuse):
        # 28 x 28 x 1 to 14 x 14 x 128
        # 第一层不加入BN
        layer1 = tf.layers.conv2d(inputs_img, 128, 3, strides=2, padding='same')
        layer1 = tf.maximum(alpha * layer1, layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)

        # 14 x 14 x 128 to 7 x 7 x 256
        layer2 = tf.layers.conv2d(layer1, 256, 3, strides=2, padding='same')
        layer2 = tf.layers.batch_normalization(layer2, training=True)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)

        # 7 x 7 x 256 to 4 x 4 x 512
        layer3 = tf.layers.conv2d(layer2, 512, 3, strides=2, padding='same')
        layer3 = tf.layers.batch_normalization(layer3, training=True)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)

        # 4 x 4 x 512 to 4*4*512 x 1
        flatten = tf.reshape(layer3, (-1, 4 * 4 * 512))
        logits = tf.layers.dense(flatten, 1)
        outputs = tf.sigmoid(logits)
        return logits, outputs

# 定义损失函数
def get_loss(inputs_real, inputs_noise, image_depth, smooth=0.1):
    """
    @param inputs_real: 输入图片，tensor类型
    @param inputs_noise: 噪声图片，tensor类型
    @param image_depth: 图片的depth（或者叫channel）
    @param smooth: label smoothing的参数
    """
    # 生成器
    g_outputs = get_generator(inputs_noise, image_depth, is_train=True)
    # 判别器
    d_logits_real, d_outputs_real = get_discriminator(inputs_real)
    d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, reuse=True)

    # 计算Loss
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                    labels=tf.ones_like(d_outputs_fake) * (1 - smooth)))

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                         labels=tf.ones_like(d_outputs_real) * (
                                                                                     1 - smooth)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                         labels=tf.zeros_like(d_outputs_fake)))
    d_loss = tf.add(d_loss_real, d_loss_fake)

    return g_loss, d_loss

# 定义梯度下降方式
def get_optimizer(g_loss, d_loss, beta1=0.4, learning_rate=0.001):
    """
    @param g_loss: Generator的Loss
    @param d_loss: Discriminator的Loss
    @learning_rate: 学习率
    """
    # 分别获取生成器和判别器的变量空间
    train_vars = tf.trainable_variables()

    g_vars = [var for var in train_vars if var.name.startswith("generator")]
    d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

    # Optimizer
    # 保证归一化先完成
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
        d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)

    return g_opt, d_opt

############################################### 设置参数 ##############################################################

batch_size = 64
noise_size = 100
epochs = 5
n_samples = 25
learning_rate = 0.001
beta1 = 0.4

############################################## 定义网络 ################################################################

data_shape = [-1,28,28,1]
inputs_real, inputs_noise = get_inputs(noise_size, data_shape[1], data_shape[2], data_shape[3])
g_loss, d_loss = get_loss(inputs_real, inputs_noise, data_shape[-1])
g_train_opt, d_train_opt = get_optimizer(g_loss, d_loss, beta1, learning_rate)

############################################### 迭代 ################################################################

# 存储训练过程中生成日志
GenLog = []
# 存储loss
losses = []
# 存储判别能力
Identify = []
# 保存生成器变量(仅保存生成器模型，保存最近5个)
saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables()
                                 if var.name.startswith("generator")],max_to_keep=5)

# 迭代次数
steps = 0
with tf.Session() as sess:
    # 获取图句柄
    graph = tf.get_default_graph()
    # 初始化全部变量
    sess.run(tf.global_variables_initializer())
    # 迭代epoch
    for e in range(epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            steps += 1
            # 提取批
            batch = mnist.train.next_batch(batch_size)
            # 变形
            batch_images = batch[0].reshape((batch_size, data_shape[1], data_shape[2], data_shape[3]))
            # scale to -1, 1
            batch_images = batch_images * 2 - 1
            # noise
            batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))

            # run optimizer
            _ = sess.run(g_train_opt, feed_dict={inputs_real: batch_images,
                                                 inputs_noise: batch_noise})
            _ = sess.run(d_train_opt, feed_dict={inputs_real: batch_images,
                                                 inputs_noise: batch_noise})
            # 每100次记录
            # 记录格式：N x batch_size x data
            #
            if steps % 101 == 0:
                # （1）记录损失函数
                train_loss_d = d_loss.eval({inputs_real: batch_images,
                                            inputs_noise: batch_noise})
                train_loss_g = g_loss.eval({inputs_real: batch_images,
                                            inputs_noise: batch_noise})
                losses.append((train_loss_d, train_loss_g))
                # （2）记录识别情况
                # d_outputs_real (discriminator/Sigmoid:0)
                identify_real  = sess.run(tf.reduce_mean(graph.get_tensor_by_name("discriminator/Sigmoid:0")),
                                          feed_dict={inputs_real: batch_images})
                # d_outputs_fake (discriminator_1/Sigmoid:0)
                identify_fake =  sess.run(tf.reduce_mean(graph.get_tensor_by_name("discriminator_1/Sigmoid:0")),
                                          feed_dict={inputs_noise: batch_noise})
                Identify.append((identify_real,identify_fake))
                # （3）记录生成样本 g_outputs (generator/Tanh:0)
                gen_samples = sess.run(graph.get_tensor_by_name("generator/Tanh:0"),
                                       feed_dict={inputs_noise: batch_noise})
                GenLog.append(gen_samples[0:26])
                #  (4) 保存生成模型
                saver.save(sess, './ckpt/generator.ckpt', global_step=steps)

                # 打印信息
                print("Epoch {}/{}...".format(e + 1, epochs),
                      'step {}...'.format(steps),
                      "Discriminator Loss: {:.4f}...".format(train_loss_d),
                      "Generator Loss: {:.4f}...".format(train_loss_g),
                      'identify_real: {:.4f}...'.format(identify_real),
                      'identify_fake: {:.4f}...'.format(identify_fake))
print('迭代补数：%d'%(steps))
# 保存信息
#  保存loss记录
with open('./trainLog/loss_variation.loss', 'wb') as l:
    losses = np.array(losses)
    pickle.dump(losses,l)
    print('保存loss信息..')
#  保存Identify
with open('./trainLog/Identify.id', 'wb') as i:
    pickle.dump(Identify,i)
    print('保存Identify信息..')
# 保存生成样本
with open('./trainLog/GenLog.log', 'wb') as g:
    pickle.dump(GenLog, g)
    print('保存GenLog信息..')