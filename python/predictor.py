import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import sys
import json
import copy

import model as M
import json_data_process as jp
import settings as parameters
from PIL import Image

class Predictor(Chain):
    def __init__(self, deepCoder):
        super(Predictor, self).__init__()
        with self.init_scope():
            self.deepCoder = deepCoder

    def __call__(self, x):
        return F.sigmoid(self.deepCoder(x))
deepCoder = M.gen_model()
serializers.load_npz("parameter.npz", deepCoder)

predictor = Predictor(deepCoder)

def print_value(value):
    s = ''
    if isinstance(value, int):
        s = "Integer " + str(value)
    else:
        s = "List "
        for x in value:
            s += str(x) + " "
    print(s)

if True:
    # Predict attribute
    f = open('./data/program_data_2.json', 'r')
    examples_ = json.load(f) # [Example]
    data = jp.preprocess_json(examples_)

    origin_image = []
    predict_image = []
    x = np.array([ele for ele,_ in data])
    y = np.array([ele for _,ele in data])
    l1 = [e for e in range(0, len(y))]
    for j in range(0,int(len(x))):
        indices = j
        train_input = x[indices]
        expected_output = y[indices]
        print(expected_output.shape)
        trained_output = predictor(np.array([train_input]))[0].data
        trained_output = np.array(trained_output)
        trained_output = np.squeeze(trained_output)
        expected_output = np.squeeze(expected_output)

        for i in range(0,parameters.attribute_width):
            origin_image.append(expected_output[i] *255)
            predict_image.append(trained_output[i] * 255)

    ori_img = Image.new('L',size=(parameters.attribute_width,int(len(origin_image)/parameters.attribute_width)))
    pre_img = Image.new('L',size=(parameters.attribute_width,int(len(predict_image)/parameters.attribute_width)))

    ori_img.putdata(origin_image)
    pre_img.putdata(predict_image)
    ori_img.save("chainer_origin.bmp")
    pre_img.save("chainer_predic.bmp")




    e = copy.deepcopy(examples_)
    examples = np.array([M.convert_example(x) for x in examples_])

    # Parse Examples
    for example in e:
        i = example["input"]
        o = example["output"]

        for value in i:
            print_value(value)
        print("---")
        print_value(o)
        print("---")

    print("---")
    # Output Attributes
    if len(sys.argv) >= 4 and sys.argv[3] == "none":
        x = 'Attribute: '
        for t in range(0, M.attribute_width):
            x += "1 "
        print(x)
    else:
        x = 'Attribute: '
        attributes = predictor(np.array([examples]))[0].data
        for t in attributes:
            x += str(t) + " "
        print(x)
else:
    # Evaluate
    embed = deepCoder.encoder.embed.valueEmbed.integerEmbed
    for l in range(0, M.integer_range + 1):
        embedInteger = embed(np.array([l], dtype=np.float32)).data[0]
        s = str(l + M.integer_min)
        for x in embedInteger:
            s += " " + str(x)
        print(s)
