from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import settings as parameters
import sys
import chainer.functions as F
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

def weight_variable(shape):
    initial = tf.random_normal(shape,stddev=tf.sqrt(1.0/shape[0]))
    return  tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.0,shape=shape)
    return  tf.Variable(initial)

session = tf.Session()

batch_size = 1
c_input = np.load('chainer/input.npy')
c_output = np.load('chainer/expect.npy')
c_tobe_embed = np.load('chainer/value_tobe_embed_x.npy')
c_split_t = np.load('chainer/value_split_t.npy')
c_split_l = np.load('chainer/value_split_l.npy')
c_actual = np.load('chainer/actual.npy')

input = tf.Variable(c_input)#(tf.int32,shape=[batch_size,parameters.example_num,parameters.input_num+1,parameters.list_length+2])


output = tf.Variable(c_output)#tf.placeholder(tf.int32,shape=[batch_size,parameters.attribute_width])


(batch, example, io_number, io_element_number) = input.shape

# 将数据的type 分割出来 ,t =[n1,2] l = [n1,n2-2]
(t, l) = tf.split(input, [2, 10], 3)
print("ValueEmbed T,L", t.shape, l.shape)


embeding = tf.Variable(tf.random_normal([parameters.integer_range + 1,parameters.embed_length]))
embed = tf.nn.embedding_lookup(embeding,tf.to_int32(l))

print("Embed",embed.shape)
shape = embed.get_shape().as_list()
embed_re = tf.reshape(embed,[-1,shape[1],shape[2],shape[3] * shape[4]])
print("Embed",embed_re.shape)
feature =  tf.concat([tf.to_float(t),embed_re],3)
print("Feature ",feature.shape)

print("ExampleEmbed feature",feature.shape)
feature_shape = feature.get_shape().as_list()

feature_re = tf.reshape(feature,[-1,feature_shape[1],feature_shape[2]*feature_shape[3]])
print("ExampleEmbed feature_re",feature_re.shape)

print("Encoder e", feature_re.shape)

feature_re_shape = feature_re.get_shape().as_list()

w_1 = weight_variable([feature_re_shape[2], parameters.hidden_layer_width])
print("w_1.shape",w_1.shape)
b_1 = bias_variable([ parameters.hidden_layer_width])

w_2 = weight_variable([ parameters.hidden_layer_width,  parameters.hidden_layer_width])
b_2 = weight_variable([ parameters.hidden_layer_width])

w_3 = weight_variable([ parameters.hidden_layer_width,  parameters.hidden_layer_width])
b_3 = weight_variable([ parameters.hidden_layer_width])
print(w_1, b_1, w_2, b_2, w_3, b_3)
features = tf.unstack(feature_re, axis=1)
after_hidden = []
for val in features:
    dense1 = tf.nn.sigmoid(tf.matmul(val, w_1) + b_1)
    dense2 = tf.nn.sigmoid(tf.matmul(dense1, w_2) + b_2)
    dense3 = tf.nn.sigmoid(tf.matmul(dense2, w_3) + b_3)
    after_hidden.append(dense3)

print(after_hidden)
after_stack = tf.stack(after_hidden, axis=1)

x1 = tf.reduce_mean(after_stack, 1)
print("Decoder x1:", x1.shape)
w1 = weight_variable([parameters.hidden_layer_width, parameters.attribute_width])
b1 = bias_variable([parameters.attribute_width])
x2 = tf.matmul(x1, w1) + b1


loss = tf.losses.sigmoid_cross_entropy( output,x2)

session.run(tf.global_variables_initializer())
input_val = session.run(input)
print(input_val.shape,c_input.shape,c_tobe_embed.shape)
print(np.array_equal(input_val,c_input))
split_t_val,split_l_val = session.run((t,l))

print(np.array_equal(np.reshape(split_t_val,[20,2]),c_split_t))
print(np.array_equal(np.reshape(split_l_val,[20,10]),c_split_l))

embeding_val = session.run(embeding)
print(embeding.shape)
embed_val = np.array(session.run(embed_re))
feature_val = session.run(feature)

x2_value = session.run(x2)
print(x2_value)
print(c_actual)

loss_val = session.run(loss)
print(loss_val)

