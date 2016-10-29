import six
import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F
import nutszebra_chainer
import functools
from collections import defaultdict
from functools import wraps


class BN_ReLU_Conv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(BN_ReLU_Conv, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(in_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False):
        return self.conv(F.relu(self.bn(x, test=not train)))

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class ResBlockWithSwapout(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, theta1, theta2, n=38):
        super(ResBlockWithSwapout, self).__init__()
        modules = []
        modules += [('bn_relu_conv1_1', BN_ReLU_Conv(in_channel, out_channel))]
        modules += [('bn_relu_conv2_1', BN_ReLU_Conv(out_channel, out_channel))]
        for i in six.moves.range(2, n + 1):
            modules.append(('bn_relu_conv1_{}'.format(i), BN_ReLU_Conv(out_channel, out_channel)))
            modules.append(('bn_relu_conv2_{}'.format(i), BN_ReLU_Conv(out_channel, out_channel)))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.n = n
        self.theta1 = theta1
        self.theta2 = theta2

    def weight_initialization(self):
        for i in six.moves.range(1, self.n + 1):
            self['bn_relu_conv1_{}'.format(i)].weight_initialization()
            self['bn_relu_conv2_{}'.format(i)].weight_initialization()

    @staticmethod
    def concatenate_zero_pad(x, h_shape, volatile, h_type):
        _, x_channel, _, _ = x.data.shape
        batch, h_channel, h_y, h_x = h_shape
        pad = chainer.Variable(np.zeros((batch, h_channel - x_channel, h_y, h_x), dtype=np.float32), volatile=volatile)
        if h_type is not np.ndarray:
            pad.to_gpu()
        return F.concat((x, pad))

    @staticmethod
    def _swapout(x, h, theta1, theta2, train=True):
        return F.dropout(x, ratio=theta1, train=train) + F.dropout(h, ratio=theta2, train=train)

    def __call__(self, x, train=False, train_dropout=True):
        h = self['bn_relu_conv1_1'](x, train=train)
        h = self['bn_relu_conv2_1'](h, train=train)
        x = ResBlockWithSwapout._swapout(ResBlockWithSwapout.concatenate_zero_pad(x, h.data.shape, h.volatile, type(h.data)), h, self.theta1[0], self.theta2[0], train=train_dropout)
        for i in six.moves.range(2, self.n + 1):
            h = self['bn_relu_conv1_{}'.format(i)](x, train=train)
            h = self['bn_relu_conv2_{}'.format(i)](h, train=train)
            x = ResBlockWithSwapout._swapout(x, h, self.theta1[i - 1], self.theta2[i - 1], train=train_dropout)
        return x

    def count_parameters(self):
        count = 0
        for i in six.moves.range(1, self.n + 1):
            count = count + self['bn_relu_conv1_{}'.format(i)].count_parameters()
            count = count + self['bn_relu_conv2_{}'.format(i)].count_parameters()
        return count


def test(func):

    @wraps(func)
    def wrapper(self, x, *args, **kwargs):
        if (self.stochastic_inference is True) and (x.volatile == 'ON' or x.volatile is True):
            y = 0
            self.stochastic_inference = False
            for i in six.moves.range(30):
                y += F.softmax(func(self, x, **kwargs))
            self.stochastic_inference = True
            return y
        else:
            return func(self, x, **kwargs)
    return wrapper


class Swapout(nutszebra_chainer.Model):

    def __init__(self, category_num, block_num=3, out_channels=(16 * 4, 32 * 4, 64 * 4), N=(10, 10, 10), Theta1=(0.0, 0.5), Theta2=(0.0, 0.5), stochastic_inference=True):
        super(Swapout, self).__init__()
        # conv
        modules = [('conv1', L.Convolution2D(3, 16, 7, 2, 3))]
        in_channel = 16
        for i, out_channel, n, theta1, theta2 in six.moves.zip(six.moves.range(1, block_num + 1), out_channels, N, Swapout.linear_schedule(Theta1[0], Theta1[1], N), Swapout.linear_schedule(Theta2[0], Theta2[1], N)):
            modules.append(('res_block_with_swapout{}'.format(i), ResBlockWithSwapout(in_channel, out_channel, theta1, theta2, n=n)))
            in_channel = out_channel
        modules.append(('bn_relu_conv', BN_ReLU_Conv(in_channel, category_num, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))))
        modules.append(('bn', L.BatchNormalization(category_num)))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.category_num = category_num
        self.block_num = block_num
        self.out_channels = out_channels
        self.N = N
        self.Theta1 = Theta1
        self.Theta2 = Theta2
        self.stochastic_inference = stochastic_inference
        self.name = 'swapout_{}_{}_{}_{}_{}_{}_{}'.format(category_num, block_num, out_channels, N, Theta1, Theta2, stochastic_inference)

    @staticmethod
    def linear_schedule(bottom_layer, top_layer, N):
        total_block = sum(N)

        def y(x):
            return (float(-1 * bottom_layer) + top_layer) / (total_block) * x + bottom_layer
        theta = []
        count = 0
        for num in N:
            tmp = []
            for i in six.moves.range(count, count + num):
                tmp.append(y(i))
            theta.append(tmp)
            count += num
        return theta

    def weight_initialization(self):
        self.conv1.W.data = self.weight_relu_initialization(self.conv1)
        self.conv1.b.data = self.bias_initialization(self.conv1, constant=0)
        for i in six.moves.range(1, self.block_num + 1):
            self['res_block_with_swapout{}'.format(i)].weight_initialization()
        self.bn_relu_conv.weight_initialization()

    @test
    def __call__(self, x, train=False, train_dropout=True):
        h = self.conv1(x)
        for i in six.moves.range(1, self.block_num + 1):
            h = self['res_block_with_swapout{}'.format(i)](h, train=train, train_dropout=train_dropout)
            h = F.max_pooling_2d(h, ksize=(3, 3), stride=(2, 2), pad=(1, 1))
        h = F.relu(self.bn(self.bn_relu_conv(h, train=train), test=not train))
        num, categories, y, x = h.data.shape
        h = F.reshape(F.average_pooling_2d(h, (y, x)), (num, categories))
        return h

    def count_parameters(self):
        count = 0
        count += functools.reduce(lambda a, b: a * b, self.conv1.W.data.shape)
        for i in six.moves.range(1, self.block_num + 1):
            count = count + self['res_block_with_swapout{}'.format(i)].count_parameters()
        count += self.bn_relu_conv.count_parameters()
        return count

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
