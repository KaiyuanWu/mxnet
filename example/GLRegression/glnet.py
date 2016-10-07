import mxnet as mx
import numpy as np
import cv2
import GLMatcher
import RelaxedL2

def build_network():
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('softmax_label')
    conv1_1 = mx.symbol.Convolution(data = data, name="conv1_1", kernel = (3,3), pad = (1,1), num_filter = 64)
    relu1_1 = mx.symbol.Activation(data = conv1_1, name='relu1_1', act_type='relu')
    conv1_2 = mx.symbol.Convolution(data = relu1_1, name="conv1_2", kernel = (3,3), pad = (1,1), num_filter = 64)
    relu1_2 = mx.symbol.Activation(data = conv1_2, name='relu1_2', act_type='relu')
    pool1 = mx.symbol.Pooling(data = relu1_2, kernel=(2,2), name="pool1",pool_type='max', stride=(2,2))

    conv2_1 = mx.symbol.Convolution(data = pool1, name="conv2_1", kernel = (3,3), pad = (1,1), num_filter = 128)
    relu2_1 = mx.symbol.Activation(data = conv2_1, name='relu2_1', act_type='relu')
    conv2_2 = mx.symbol.Convolution(data = relu2_1, name="conv2_2", kernel = (3,3), pad = (1,1), num_filter = 128)
    relu2_2 = mx.symbol.Activation(data = conv2_2, name='relu2_2', act_type='relu')
    pool2 = mx.symbol.Pooling(data = relu2_2, kernel=(2,2), name="pool2",pool_type='max', stride=(2,2))

    conv3_1 = mx.symbol.Convolution(data = pool2, name="conv3_1", kernel = (3,3), pad = (1,1), num_filter = 256)
    relu3_1 = mx.symbol.Activation(data = conv3_1, name='relu3_1', act_type='relu')
    conv3_2 = mx.symbol.Convolution(data = relu3_1, name="conv3_2", kernel = (3,3), pad = (1,1), num_filter = 256)
    relu3_2 = mx.symbol.Activation(data = conv3_2, name='relu3_2', act_type='relu')
    conv3_3 = mx.symbol.Convolution(data = relu3_1, name="conv3_3", kernel = (3,3), pad = (1,1), num_filter = 256)
    relu3_3 = mx.symbol.Activation(data = conv3_3, name='relu3_3', act_type='relu')
    pool3 = mx.symbol.Pooling(data = relu3_3, kernel=(2,2), name="pool2",pool_type='max', stride=(2,2))

    
    conv4_1 = mx.symbol.Convolution(data = pool3, name="conv4_1", kernel = (3,3), pad = (1,1), num_filter = 512)
    relu4_1 = mx.symbol.Activation(data = conv4_1, name='relu4_1', act_type='relu')
    conv4_2 = mx.symbol.Convolution(data = relu4_1, name="conv4_2", kernel = (3,3), pad = (1,1), num_filter = 512)
    relu4_2 = mx.symbol.Activation(data = conv4_2, name='relu4_2', act_type='relu')
    conv4_3 = mx.symbol.Convolution(data = relu4_2, name="conv4_3", kernel = (3,3), pad = (1,1), num_filter = 512)
    relu4_3 = mx.symbol.Activation(data = conv4_3, name='relu4_3', act_type='relu')
    pool4 = mx.symbol.Pooling(data = relu4_3, kernel=(2,2), name="pool4",pool_type='max', stride=(2,2))

    fc0_landmarks = mx.symbol.FullyConnected(data = pool4, name="fc0_landmarks", num_hidden = 256)
    sigmoid0_landmarks = mx.symbol.Activation(data = fc0_landmarks, name = "sigmoid0_landmarks", act_type='sigmoid')

    landmarks_global = mx.symbol.FullyConnected(data = sigmoid0_landmarks, name="landmarks_global", num_hidden = 10)

    conv3_3_dense_prediction = mx.symbol.Convolution(data = relu3_3, name="conv3_3_dense_prediction", kernel = (3,3), pad = (1,1), num_filter = 10)
    dense_prediction = mx.symbol.Activation(data = conv3_3_dense_prediction, name = 'dense_prediction', act_type='tanh')

    gl_loss = mx.symbol.Custom(global_pred=landmarks_global, local_pred=dense_prediction, label=label, op_type='glmatcher', name='gl_loss')
    g_loss = mx.symbol.Custom(predict = landmarks_global, label = label,  op_type='relaxedl2', name='g_loss')

    net = mx.symbol.Group([gl_loss, g_loss]) 
    return net
    
 class Multi_RMSE(mx.metric.EvalMetric):
    def __init__(self, num=None):
        super(Multi_RMSE, self).__init__('multi-rmse', num)

    def update(self, labels, preds):
        label = labels[0].asnumpy()
        for i in range(len(preds)):
            pred_label = preds[i].asnumpy()
	    ex = pred_label[...,1]-pred_label[...,2]
	    ey = pred_label[...,6]-pred_label[...,7]
	    eye_dist = ex*ex+ey*ey
	    dist = pred_label - label
	    
	    dist = ((dist*dist).sum(axis=1)/eye_dist)**0.5
            self.sum_metric[i] += dist.sum()
            self.num_inst[i] += pred_label.shape[0]
batch_size = 100
num_epochs = 160
device = mx.gpu(0)
lr = 0.001
train = mx.io.ImageRecordIter( path_imgrec = '/home/kaiwu/FaceAlignment/data/test.rec',
                               data_shape=(3,80,80), batch_size = batch_size, label_width = 10,
                               rand_crop = False, rand_mirror = False, shuffle = False,prefetch_buffer = 3,preprocess_threads=2,
                               mean_r = 127.5, mean_g  = 127.5, mean_b = 127.5, scale = 0.0078125 )

network = build_network()     
epoch_size = 200000/batch_size
model_args = {}
model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step = max(int(epoch_size * 40), 1),
            factor = 0.1)
model = mx.model.FeedForward(
    ctx = device,
    symbol = network,
    num_epoch = num_epochs,
    momentum           = 0.9,
    wd                 = 0.00001,
    initializer = mx.init.Xavier(factor_type = 'in', magnitude = 1.), **model_args)
odel.fit(X = train, eval_metric = Multi_RMSE(num=2),
          batch_end_callback = mx.callback.Speedometer(batch_size, 100), epoch_end_callback=mx.callback.do_checkpoint('models/face'))    
