#Example: python init_model.py  center_loss_sym 1 128 128
import mxnet as mx
import numpy as np
import sys

print("import", sys.argv[1])
sym_modul = __import__(sys.argv[1])
sym = sym_modul.get_network()

xavier_initilializer = mx.init.Xavier(rnd_type="uniform", factor_type="in", magnitude=2.34)
gaussian_initilializer = mx.init.Normal(sigma = 0.01)

init = mx.init.Mixed( patterns=[ '.*resb' , '.*' ] , initializers = [xavier_initilializer, gaussian_initilializer] )

c = int(sys.argv[2])
h = int(sys.argv[3])
w = int(sys.argv[4])

input_shapes = {'data':(1, c, h, w)}
exe = sym.simple_bind(ctx=mx.cpu(), **input_shapes)
arg_arrays = exe.arg_dict
for name, arr in arg_arrays.items():
        if name not in input_shapes:
            init(name, arr)
#Check std of the params of each layer
for name in  exe.arg_dict:
	print(name, "shape", exe.arg_dict[name].shape, "std: ", np.std(exe.arg_dict[name].asnumpy().reshape(-1)))

mx.model.save_checkpoint(sys.argv[1], 1, sym, exe.arg_dict, exe.aux_dict)
