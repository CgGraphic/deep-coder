from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import settings as parameters
import sys

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

class IntergerEmbed(object):
    # Embedding set
    def __init__(self,integer_range,embed_length):
        self.embeding = tf.Variable(tf.random_normal([integer_range + 1,embed_length]))
        print(self.embeding)
    # Call Embedding
    def __call__(self,x):
        return tf.nn.embedding_lookup(self.embeding,x)

class ValueEmbed(object):
    def __init__(self,integer_range,embed_length):
        self.interger_embed = IntergerEmbed(integer_range,embed_length)

    # Input: [Value([typed-vector (valueID)*])]
    # 类别矩阵 和 输入的Vector
    # 前两个是类别 后面是一个最长的10的输入List
    # Onput [[double+]]
    # 查找Embed之后的数据
    def __call__(self, x):
        with tf.name_scope('ValueEmbed'):
            # n1 数据尺寸
            (batch,example,io_number,io_element_number) = x.shape
            # 将数据的type 分割出来 ,t =[n1,2] l = [n1,n2-2]
            (t,l) = tf.split(x,[2,10],3)
            print("ValueEmbed T,L",t.shape,l.shape)
            embed = self.interger_embed(l)

            variable_summaries(embed)

            print("Embed",embed.shape)
            shape = embed.get_shape().as_list()
            embed = tf.reshape(embed,[-1,shape[1],shape[2],shape[3] * shape[4]])
            print("Embed",embed.shape)
            feature =  tf.concat([tf.to_float(t),embed],3)
            print("Feature ",feature.shape)
            return  feature

class ExampleEmbed(object):
    def __init__(self,input_num,integer_range,embed_length):
        self.valueEmbed = ValueEmbed(integer_range,embed_length)
    # Input： 一共有n1个Exampl 每一个Exmaple 由n2 个输入输出组成，每一个输入输出有n3个值
    def __call__(self, x):
        with tf.name_scope('ExampleEmbed'):
            (batch,example,io_number,io_element_number) = x.shape
            x1 = x
            print("ExampleEmbed x1",x1.shape)
            x_ =self.valueEmbed(x1)
            print("ExampleEmbed x_",x_.shape)
            shape = x_.get_shape().as_list()

            x_ = tf.reshape(x_,[-1,shape[1],shape[2]*shape[3]])
            print("ExampleEmbed x_",x_.shape)

            return x_

def weight_variable(shape):
    initial = tf.random_normal(shape,stddev=tf.sqrt(1.0/shape[0]))
    return  tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.0,shape=shape)
    return  tf.Variable(initial)

class Encoder(object):
    def __init__(self,embed,out_width):
        self.embed = embed
        self.units = out_width
    def __call__(self,x):
        with tf.name_scope('Encoder'):
            e = self.embed(x)
            print("Encoder e",e.shape)

            shape = e.get_shape().as_list()

            w_1 = weight_variable([shape[2],self.units])
            b_1 = bias_variable([self.units])


            w_2 = weight_variable([self.units,self.units])
            b_2 = weight_variable([self.units])

            w_3 = weight_variable([self.units,self.units])
            b_3 = weight_variable([self.units])
            print(w_1,b_1,w_2,b_2,w_3,b_3)
            features = tf.unstack(e,axis=1)
            after_hidden = []
            for val in features:
                dense1 = tf.nn.sigmoid(tf.matmul(val,w_1) + b_1)
                dense2 = tf.nn.sigmoid(tf.matmul(dense1,w_2) + b_2)
                dense3 = tf.nn.sigmoid(tf.matmul(dense2,w_3) + b_3)
                after_hidden.append(dense3)

            print(after_hidden)
            after_stack = tf.stack(after_hidden,axis=1)

            variable_summaries(w_1)
            variable_summaries(b_1)
            variable_summaries(w_2)
            variable_summaries(b_2)
            variable_summaries(w_3)
            variable_summaries(b_3)

            #e = tf.reshape(e,[-1,shape[2]])
            # print("Encoder e",e.shape)
            # dense1 = tf.contrib.layers.fully_connected(inputs=e,num_outputs=self.units,activation_fn=tf.nn.sigmoid)
            # print("Dense1",dense1.shape)
            # dense2 = tf.contrib.layers.fully_connected(inputs=e,num_outputs=self.units,activation_fn= tf.nn.sigmoid)
            # print("Dense1",dense2.shape)
            # dense3 = tf.contrib.layers.fully_connected(inputs=e,num_outputs=self.units,activation_fn= tf.nn.sigmoid)




            return  after_stack

class Decoder(object):
    def __init__(self,out_width):
        self.units = out_width
    def __call__(self, x):
        with tf.name_scope('Decoder'):
            shape = x.get_shape().as_list()
            print("Decoder x:",x.shape)
            x1 = tf.reduce_mean(x,1)
            print("Decoder x1:",x1.shape)
            w1 = weight_variable([parameters.hidden_layer_width,self.units])
            b1 = bias_variable([self.units])
            x2 = tf.matmul(x1,w1) + b1

            variable_summaries(w1)
            variable_summaries(b1)
            print(w1,b1,x2)
            #x2 = tf.contrib.layers.fully_connected(inputs=x1,num_outputs=self.units,activation_fn=None)
            return  x2

class DeepCoder:
    def __init__(self,encoder,decoder,example_number):
        self.encoder = encoder
        self.decoder = decoder
        self.example_number = example_number
    def __call__(self,x):
        (batch_number,example_num,io_number,io_ele_number) = x.shape
        print("DeepCoder x",x.shape)
        x1 = self.encoder(x)
        print("DeepCoder x",x1.shape)

        x2 = self.decoder(x1)
        return x2



def gen_model():
    embed = ExampleEmbed(parameters.input_num, parameters.integer_range, parameters.embed_length)
    encoder = Encoder(embed, parameters.hidden_layer_width)
    decoder = Decoder(parameters.attribute_width)
    deepCoder = DeepCoder(encoder, decoder,parameters.example_num)
    return deepCoder


# ones = tf.placeholder(tf.float32,[None,3,5])
# trans = tf.ones([5,4])
#
# features = tf.unstack(ones,axis=1)
# after_hidden = []
# for val in features:
#     after_hidden.append(tf.matmul(val,trans))
# after_stack = tf.stack(after_hidden,axis=1)
# print(after_stack)
#
# print("Split")
# split,split2 = tf.split(ones,[1,2],1)
#
# sess = tf.Session()
#
# ones_vale = [[[1,2,3,4,5],
#               [6,7,8,9,10],
#               [11,12,13,14,15]],
#              [[16,17,18,19,20],
#               [21,22,23,24,25],
#               [26,27,28,29,30]]
# ]
#
# s1,s2 =sess.run([split,split2],{ones:ones_vale})
#
# print(s1)
# print("\nhhh\n")
# print(s2)
np.set_printoptions(threshold=sys.maxsize)
expected = np.load('expect.npy')
actual = np.load('actual.npy')

cross = tf.losses.sigmoid_cross_entropy(expected,actual)
sess = tf.Session()
print(cross.eval(session=sess))







