import caffe
import numpy as np
if __name__ == '__main__':
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver=caffe.AdamSolver('./cmp_adam_solver.prototxt')
    solver.restore('./save/cmp_iter_100000.solverstate')
    for i in range(10000):
        solver.step(10)
        print solver.net.blobs['loss'].data
