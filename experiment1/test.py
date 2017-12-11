import caffe
import numpy as np
import time
if __name__ == '__main__':
    caffe.set_device(0)
    caffe.set_mode_gpu()
    #net=caffe.Net('./cmp_test_imp_128.prototxt','./save/imp/5.caffemodel',caffe.TEST)
    #net=caffe.Net('./cmp_test_imp.prototxt','./save/imp/3.caffemodel',caffe.TEST)
    #net=caffe.Net('./cmp_test_128.prototxt','./save/cmp/5.caffemodel',caffe.TEST)
    net=caffe.Net('./model/cmp_test.prototxt','./save/cmp/3.caffemodel',caffe.TEST)
    rate=0
    net.forward()
    st_time=time.time()
    for i in range(10):
        net.forward()
        rate+=net.blobs['loss'].data
        print net.blobs['loss'].data
    end_time=time.time()
    print 'time used for 24 pics:%.4f'%((end_time-st_time)/10.0)
    print rate/24.0
