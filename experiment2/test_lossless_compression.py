import numpy as np
import caffe
import cv2
import math
if __name__ == '__main__':
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net=caffe.Net('./lossless_cmp_v1_deploy.prototxt','./save/cmp_iter_110000.caffemodel',caffe.TEST)
    f=open('test.txt','r')
    flist=[]
    for pt in f.readlines():
        flist.append(pt[:-1])
    f.close()
    mrate=0
    idx=0
    num_p=len(flist)
    ytrans=lambda x:0.299*x[2]+0.587*x[1]+0.114*x[0]
    for pimg in flist[0:num_p]:
       print pimg
       img=cv2.imread(pimg)
       if img.shape[0] % 8 >0:
           img=img[0:img.shape[0]-img.shape[0]%8,:]
       if img.shape[1] % 8 >0:
           img=img[:,0:img.shape[1]-img.shape[1]%8]  
       net.blobs['data'].reshape(1,1,img.shape[0],img.shape[1])
       data=ytrans(img.transpose(2,0,1)).astype(np.uint8)
       net.blobs['data'].data[0]=data.astype(np.float32)
       net.forward()
       print net.blobs['loss'].data
       mrate+=net.blobs['loss'].data
    print mrate/num_p