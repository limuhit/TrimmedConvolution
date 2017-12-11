import caffe
import numpy as np
if __name__ == '__main__':
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver=caffe.AdamSolver('./cmp_adam_solver.prototxt')
    solver.restore('./save/cmp_iter_10000.solverstate')
    #solver.net.copy_from('./save/cmp_iter_60000.caffemodel')
    for i in range(10000):
        solver.step(10)
        print solver.net.blobs['loss'].data
