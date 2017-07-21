import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import settings as parameters

print_dim = False

class IntegerEmbed(Chain):
    def __init__(self, integer_range, embed_length):
        super(IntegerEmbed, self).__init__()
        with self.init_scope():
            self.id = L.EmbedID(integer_range + 1, embed_length)
    def __call__(self, x):
        # Input : [ID ([0, ..., integer_range-1, NULL(integer_range)])]
        # Output: [[float]^embed_length]
        if print_dim:
            print("IntegerEmbed: x ",x.shape)
        return self.id(F.cast(x, np.int32))

class ValueEmbed(Chain):
    def __init__(self, integer_range, embed_length):
        super(ValueEmbed, self).__init__()
        with self.init_scope():
            self.integerEmbed = IntegerEmbed(integer_range, embed_length)
    def __call__(self, x):
        # Input: [Value([type-vector (valueID)*])]
        # Output: [[doublshapee+]]
        (n1, n2) = x.shape
        if print_dim:
            print("ValueEmbed: n1,n2 ",n1,n2)
        (t, l) = F.split_axis(x, [2], 1)
        if print_dim:
            print("ValueEmbed: t,l ",t.shape,l.shape)
        embedL_ = self.integerEmbed(l)
        if print_dim:
            print("ValueEmbed: embedL_ ",embedL_.shape)
        embedL = embedL_.reshape([n1, int(embedL_.size / n1)])
        if print_dim:
            print("ValueEmbed: embedL_ ",embedL_.shape)
        return F.concat((t, embedL))

class ExampleEmbed(Chain):
    def __init__(self, input_num, integer_range, embed_length):
        super(ExampleEmbed, self).__init__()
        with self.init_scope():
            self.valueEmbed = ValueEmbed(integer_range, embed_length)
    def __call__(self, x):
        # Input: [Example([Value])]
        # Output: [[double+]]
        # n1 一共多少输入输出对 n2 每一个输入输出中一共有多少数据  input_num + 1(output_number), n3 每一个数据的维度
        (n1, n2, n3) = x.shape
        if print_dim:
            print("ExampleEmbed: n1,n2,n3 ",n1,n2,n3)
        x1 = x.reshape(n1 * n2, n3)
        if print_dim:
            print("ExampleEmbed: x1 ",x1.shape)
        x_ = self.valueEmbed(x1)
        if print_dim:
            print("ExampleEmbed: x_ ",x_.shape,x_.size)
        return x_.reshape([n1, int(x_.size / n1)])

class Encoder(Chain):
    def __init__(self, embed, out_width):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.embed = embed
            self.h1 = L.Linear(None, out_width)
            self.h2 = L.Linear(None, out_width)
            self.h3 = L.Linear(None, out_width)
    def __call__(self, x):
        # Input: [Example]
        # Output: [[double+]]
        e = self.embed(x)
        if print_dim:
            print("Encoder",e.shape)
        x1 = F.sigmoid(self.h1(e))
        if print_dim:
            print(x1.shape)
        x2 = F.sigmoid(self.h2(x1))
        x3 = F.sigmoid(self.h2(x2))
        return x3

class Decoder(Chain):
    def __init__(self, out_width):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.h = L.Linear(None, out_width)
    def __call__(self, x):
        # Input: [[[double+]]]
        # Output: [[double+](attribute before sigmoid)]
        x1 = F.average(x, axis=1)
        x2 = self.h(x1)
        return x2

class DeepCoder(Chain):
    def __init__(self, encoder, decoder, example_num):
        super(DeepCoder, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.decoder = decoder
            self.example_num = example_num
    def __call__(self, x):
        # Input: [[Example]]
        # Output: [[double]+(attribute before sigmoid)]
        (data_num, example_num, n3, n4) = x.shape
        if print_dim:
            print("DeepCoder x",x.shape)
        x1 = self.encoder(x.reshape(data_num * example_num, n3, n4))
        (_, vec_length) = x1.shape
        x2 = self.decoder(x1.reshape(data_num, example_num, vec_length))
        if print_dim:
            print("DeepCoder x2",x2.shape,x1.shape)
        return x2



def gen_model():
    embed = ExampleEmbed(parameters.input_num, parameters.integer_range, parameters.embed_length)
    encoder = Encoder(embed, parameters.hidden_layer_width)
    decoder = Decoder(parameters.attribute_width)
    deepCoder = DeepCoder(encoder, decoder,parameters.example_num)
    return deepCoder


