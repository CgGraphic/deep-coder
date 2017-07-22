import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import json
import sys
import traceback
import model as M
import json_data_process as jp
import settings as parameters
from PIL import Image

class Model(Chain):
    def __init__(self, deepCoder):
        super(Model, self).__init__()
        with self.init_scope():
            self.deepCoder = deepCoder
    def __call__(self, input_, output):
        if M.print_dim:
            print(input_.shape)
            np.save('chainer/input.npy',input_)
        actual = self.deepCoder(Variable(input_))
        expected = Variable(output)

        loss = F.sigmoid_cross_entropy(actual, expected)
        if M.print_dim:
            act_ar =  np.array(actual.data)
            exp_ar = np.array(expected.data)

            print(exp_ar.dtype,exp_ar.shape)
            print(act_ar.dtype,act_ar.shape)
            np.save('chainer/actual.npy',act_ar)
            np.save('chainer/expect.npy',exp_ar)
        report({'loss': loss}, self)
        return loss

class Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, data):
        super(chainer.dataset.DatasetMixin, self).__init__()
        self.data = data
    def __len__(self):
        return len(self.data)
    def get_example(self, i):
        return self.data[i]

deepCoder = M.gen_model()
model = Model(deepCoder)

file_name = "./data/program_data_2.json"
store_file = "parameter.npz"

f = open(file_name, 'r')
x = json.load(f)
data = jp.preprocess_json(x)



np.set_printoptions(threshold=sys.maxsize)

l1 = [e for e in range(0, len(data)) if e % 100 != 0]
l2 = [e for e in range(0, len(data)) if e % 100 == 0]

train = Dataset([data[e] for e in l1])
test = Dataset([data[e] for e in l2])

try:
    train_iter = iterators.SerialIterator(train, batch_size=50, shuffle=True)
    test_iter = iterators.SerialIterator(test, batch_size=50, repeat=False, shuffle=False)

    deepCoder = M.gen_model()
    model = Model(deepCoder)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (50, 'epoch'), out='result')

    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()
    serializers.save_npz(store_file, deepCoder)



except:
    print(traceback.format_exc())

