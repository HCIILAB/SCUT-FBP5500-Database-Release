#!/usr/bin/env python
import random
import skimage
import numpy as np
import matplotlib.pyplot as plt
import caffe

def get_mean_npy(mean_bin_file, crop_size=None, isColor = True):
    mean_blob = caffe.proto.caffe_pb2.BlobProto()
    mean_blob.ParseFromString(open(mean_bin_file, 'rb').read())

    mean_npy = caffe.io.blobproto_to_array(mean_blob)
    _shape = mean_npy.shape
    mean_npy = mean_npy.reshape(_shape[1], _shape[2], _shape[3])

    if crop_size:
        mean_npy = mean_npy[
            :,
            (_shape[2] - crop_size[0]) / 2:(_shape[2] + crop_size[0]) / 2,
            (_shape[3] - crop_size[1]) / 2:(_shape[3] + crop_size[1]) / 2]

    if not isColor:
    	mean_npy = np.mean(mean_npy, axis = 0)
    	mean_npy = mean_npy[np.newaxis, :, :]

    return mean_npy

def crop_img(img, crop_size, crop_type='center_crop'):
    '''
        crop_type is one of 'center_crop',
                            'random_crop', 'random_size_crop'
    '''

    if crop_type == 'center_crop':
        sh = crop_size 
        sw = crop_size
        hh = (img.shape[0] - sh) / 2
        ww = (img.shape[1] - sw) / 2
    elif crop_type == 'random_crop':
        sh = crop_size
        sw = crop_size
        hh = random.randint(0, img.shape[0] - sh)
        ww = random.randint(0, img.shape[1] - sw)
    elif crop_type == 'random_size_crop':
        sh = random.randint(crop_size[0], img.shape[0])
        sw = random.randint(crop_size[1], img.shape[1])
        hh = random.randint(0, img.shape[0] - sh)
        ww = random.randint(0, img.shape[1] - sw)
    img = img[hh:hh + sh, ww:ww + sw]
    if crop_type == 'random_size_crop':
        img = skimage.transform.resize(img, crop_size, mode='reflect')    
    return img


def load_img(path, resize=128, isColor=True,
             crop_size=112, crop_type='center_crop',
             raw_scale=1, means=None):
    '''
        crop_type is one of None, 'center_crop',
                            'random_crop', 'random_size_crop'
    '''
    img = skimage.io.imread(path)

    if resize is not None and img.shape != resize:
        img = skimage.transform.resize(img, resize, mode='reflect')
    if crop_size and crop_type:
        img = crop_img(img, crop_size, crop_type)
    if isColor:
        img = skimage.color.gray2rgb(img)
        img = img.transpose((2, 0, 1))
        img = img[(2, 1, 0), :, :]
    else:
        img = skimage.color.rgb2gray(img)
        img = img[np.newaxis, :, :]

    img = skimage.img_as_float(img).astype(np.float32) * raw_scale #img is between[0,1]

    if means is not None:
        if means.ndim == 1 and isColor:
            means = means[:, np.newaxis, np.newaxis]
        img -= means
        img = img / 255
    return img


def main():
    deploy = ['./alexnet_deploy.prototxt', './resnet18_deploy.prototxt', './resnext50_deploy.prototxt']
    models = ['./models/alexnet.caffemodel', './models/resnet18.caffemodel', './models/resnext50.caffemodel']
    # 227 * 227 for alexnet, 224 * 224 for resnet18 and resnext50
    shapes = [(1, 3, 227, 227), (1, 3, 224, 224)] 

    network_file = deploy[1]
    pretrained_model = models[1]
    batch_shape = shapes[1]
    is_color = True

    mean_file = "../data/1/256_train_mean.binaryproto"
    means = get_mean_npy(mean_file, crop_size = batch_shape[2:], isColor = is_color)
    roots = '../data/faces/'
    file = open('../data/1/test_1.txt','r')
    lines = file.readlines()
    file.close()
    
    caffe.set_mode_gpu()
    net = caffe.Net(network_file, pretrained_model, caffe.TEST)  # set caffe model
    labellist = []
    preclist = []

    for line in lines:
        linesplit = line.strip().split(' ')
        filename = linesplit[0]
        label = float(linesplit[1])
    	imgdir = roots + filename
    	inputs = load_img(imgdir, resize = (256, 256), isColor = is_color, crop_size = batch_shape[3], \
    		crop_type = 'center_crop', raw_scale = 255, means = means)
    	net.blobs['data'].data[...] = inputs
    	prec = net.forward().values()[0][0][0]
        labellist.append(label)
        preclist.append(prec)
        print filename

    labellist = np.array(labellist)
    preclist = np.array(preclist)
    correlation = np.corrcoef(labellist, preclist)[0][1]
    mae = np.mean(np.abs(labellist - preclist))
    rmse = np.sqrt(np.mean(np.square(labellist - preclist)))

    print('Correlation {corr:.4f}\t'
            'Mae {mae:.4f}\t'
            'Rmse {rmse:.4f}\t'.format(corr=correlation, mae=mae, rmse=rmse))


if __name__ == '__main__':
    main()
