import numpy as np
import lmdb
import caffe
import cv2
import os
'''
test: bool flag: create dataset for test or train
idx: specify the model used for extract binary codes
channel_128: the output channel is 128 or 64 
'''
def create_lmdb(net,test=False,idx=2,channel_128=False):
    if test:
        num_batch=24
        dataset_name='f:/compress/old_%d_imp_test_lmdb'%idx
    else: 
        num_batch=1000
        dataset_name='f:/compress/old_%d_imp_lmdb'%idx
    map_size=np.prod( net.blobs['imp_conv2'].data.shape)*num_batch*1.6
    env = lmdb.open(dataset_name,map_size)
    i = 0
    datum=caffe.proto.caffe_pb2.Datum()
    shape=net.blobs['imp_conv2'].data.shape
    datum.channels=shape[1]
    datum.height=shape[2]
    datum.width=shape[3]
    with env.begin(write=True) as txn:
        for li in range(num_batch):
            net.forward()
            if channel_128:
                plist=((net.blobs['imp_conv2'].data+0.00001)*32+1).astype(np.uint8)
            else:
                plist=((net.blobs['imp_conv2'].data+0.00001)*16+1).astype(np.uint8)
            print "%dth batch"%(li+1)
            for pidx in range(shape[0]):
                 datum.data=plist[pidx].tobytes()
                 datum.label=int(1)
                 stri_id='{:08}'.format(i)
                 i = i+1
                 txn.put(stri_id.encode('ascii'),datum.SerializePartialToString())
                 if i % 100 == 0:
                    print i
if __name__ == '__main__':
    idx=5
	if idx>4:
		net=caffe.Net('./extract_binary_data/compress_v1_extract_data_128.prototxt','./extract_binary_data/%d.caffemodel'%idx,caffe.TEST)
	else:
		net=caffe.Net('./extract_binary_data/compress_v1_extract_data.prototxt','./extract_binary_data/%d.caffemodel'%idx,caffe.TEST)
    create_lmdb(net,False,idx,False)

