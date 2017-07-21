import sys
import tensorflow as tf
import numpy as np
import json_data_process as jp
import json
import tensor_flow_model as tf_model
import settings as parameters
from random import shuffle
from PIL import Image


class Model(object):
    def __init__(self,deepcoder):
        self.deepcoder = deepcoder
        self.input = tf.placeholder(tf.int32,shape=[100,parameters.example_num,parameters.input_num+1,parameters.list_length+2])
        self.output = tf.placeholder(tf.int32,shape=[100,parameters.attribute_width])
        self.output = tf.to_float(self.output)
        self.traind_output = self.deepcoder(self.input)

        self.predictor_out=  tf.sigmoid(self.traind_output)
        print(self.traind_output)
        print(self.output)
        self.loss = tf.losses.sigmoid_cross_entropy( self.traind_output,self.output)

        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)








deepCoder = tf_model.gen_model()
model = Model(deepCoder)

file_name = "./data/program_data_2.json"
store_file = "parameter.npz"

f = open(file_name, 'r')
f = json.load(f)
data = jp.preprocess_json(f)
x = np.array([ele for ele,_ in data])
y = np.array([ele for _,ele in data])

l1 = np.array([e for e in range(0, len(y)) if e % 100 != 0])
l2 = np.array([e for e in range(0, len(y)) if e % 100 == 0])

np.set_printoptions(threshold=sys.maxsize)
session = tf.Session()
session.run(tf.global_variables_initializer())
for i in range(0,parameters.epoch):
    shuffle(l1)

    total_cost = 0
    for j in range(0,int(len(l1)/100)-1):
        indices = l1[j*100:min((j+1)*100,len(l1))]
        train_input = x[indices]
        output_expected = y[indices]
        _,trained_output,expected_output,loss = session.run([model.train_step,model.traind_output,model.output,model.loss],{model.input :train_input,model.output : output_expected})
        #print(cost)
        total_cost += loss
        #if j == 0:
            #print(trained_output)
            #print(expected_output)
    print("Epoch: ",i,"Cost: ",total_cost/len(l1))


## Begin Predictor

origin_image = []
predict_image = []
l1 = [e for e in range(0, len(y))]
for j in range(0,int(len(x)/100)-1):
    indices = l1[j*100:min((j+1)*100,len(l1))]
    train_input = x[indices]
    expected_output = y[indices]
    print(expected_output.shape)
    trained_output = session.run([model.predictor_out],{model.input:train_input})
    trained_output = np.array(trained_output)
    trained_output = np.squeeze(trained_output)
    expected_output = np.squeeze(expected_output)
    for  k in range(0,100):
        for i in range(0,parameters.attribute_width):
            origin_image.append(expected_output[k][i] *255)
            predict_image.append(trained_output[k][i] * 255)

ori_img = Image.new('L',size=(parameters.attribute_width,int(len(origin_image)/parameters.attribute_width)))
pre_img = Image.new('L',size=(parameters.attribute_width,int(len(predict_image)/parameters.attribute_width)))

ori_img.putdata(origin_image)
pre_img.putdata(predict_image)
ori_img.save("tf_origin.bmp")
pre_img.save("tf_predic.bmp")



