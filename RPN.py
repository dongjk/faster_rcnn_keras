import os
import traceback
import numpy as np
import numpy.random as npr
import keras.backend as K
from keras.layers import Conv2D, BatchNormalization
from keras.models import Input, Model
from keras.applications import InceptionResNetV2
from keras.preprocessing.image import load_img, img_to_array
from utils import generate_anchors, draw_anchors, bbox_overlaps, bbox_transform,\
                    loss_cls, smoothL1, parse_label, unmap

k=9 #anchor number for each point
##################  RPN Model  #######################
feature_map_tile = Input(shape=(None,None,1536))
convolution_3x3 = Conv2D(
    filters=2048,
    kernel_size=(3, 3),
    padding='same',
    name="3x3"
)(feature_map_tile)

convolution_3x3=BatchNormalization()(convolution_3x3)

output_deltas = Conv2D(
    filters= 4 * k,
    kernel_size=(1, 1),
    activation="linear",
    kernel_initializer="uniform",
    name="deltas1"
)(convolution_3x3)

output_scores = Conv2D(
    filters=1 * k,
    kernel_size=(1, 1),
    activation="sigmoid",
    kernel_initializer="glorot_normal",
    name="scores1"
)(convolution_3x3)
model = Model(inputs=[feature_map_tile], outputs=[output_scores, output_deltas])
model.compile(optimizer='rmsprop', loss={'deltas1':smoothL1, 'scores1':loss_cls})

##################  prepare batch  #######################
BG_FG_FRAC=3

#load an example to void graph problem
#TODO fix this.
# pretrained_model = InceptionResNetV2(include_top=False)
# img=load_img("./ILSVRC2014_train_00010391.JPEG")
# x = img_to_array(img)
# x = np.expand_dims(x, axis=0)
# not_used=pretrained_model.predict(x)

def produce_batch(filepath, gt_boxes, h_w):
    feature_map=np.load(filepath)['fc']
    # feature_map=feature_map.repeat(3, axis=1).repeat(3, axis=2)
    height = np.shape(feature_map)[1]
    width = np.shape(feature_map)[2]
    num_feature_map=width*height
    #calculate output w, h stride
    w_stride = h_w[1] / width
    h_stride = h_w[0] / height
    #generate base anchors according output stride.
    #base anchors are 9 anchors wrt a tile (0,0,w_stride-1,h_stride-1)
    base_anchors=generate_anchors(w_stride,h_stride)
    #slice tiles according to image size and stride.
    #each 1x1x1532 feature map is mapping to a tile.
    shift_x = np.arange(0, width) * w_stride
    shift_y = np.arange(0, height) * h_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), 
                            shift_y.ravel())).transpose()
    #apply base anchors to all tiles, to have a num_feature_map*9 anchors.
    all_anchors = (base_anchors.reshape((1, 9, 4)) +
                    shifts.reshape((1, num_feature_map, 4)).transpose((1, 0, 2)))
    total_anchors = num_feature_map*9
    all_anchors = all_anchors.reshape((total_anchors, 4))
    #only keep anchors inside image+borader.
    border=10
    inds_inside = np.where(
            (all_anchors[:, 0] >= -border) &
            (all_anchors[:, 1] >= -border) &
            (all_anchors[:, 2] < h_w[1]+border ) &  # width
            (all_anchors[:, 3] < h_w[0]+border)    # height
    )[0]
    anchors=all_anchors[inds_inside]
    # calculate overlaps each anchors to each gt boxes, 
    # a matrix with shape [len(anchors) x len(gt_boxes)]
    overlaps = bbox_overlaps(anchors, gt_boxes)
    # find the gt box with biggest overlap to each anchors, 
    # and the overlap ratio. result (len(anchors),)
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    # find the anchor with biggest overlap to each gt boxes, 
    # and the overlap ratio. result (len(gt_boxes),)
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    #labels, 1=fg/0=bg/-1=ignore
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)
    # set positive label, define in Paper3.1.2: 
    # We assign a positive label to two kinds of anchors: (i) the
    # anchor/anchors with the highest Intersection-overUnion
    # (IoU) overlap with a ground-truth box, or (ii) an
    # anchor that has an IoU overlap higher than 0.7 with any gt boxes
    labels[max_overlaps >= .7] = 1
    labels[gt_argmax_overlaps] = 1
    labels[max_overlaps <= .3] = 0
    # set negative labels
    # subsample positive labels if we have too many
#     num_fg = int(RPN_FG_FRACTION * RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
#     if len(fg_inds) > num_fg:
#         disable_inds = npr.choice(
#             fg_inds, size=(len(fg_inds) - num_fg), replace=False)
#         labels[disable_inds] = -1
    # subsample negative labels if we have too many
    num_bg = int(len(fg_inds) * BG_FG_FRAC)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
    #
    batch_inds=inds_inside[labels!=-1]
    batch_inds=(batch_inds / k).astype(np.int)
    full_labels = unmap(labels, total_anchors, inds_inside, fill=-1)
    batch_label_targets=full_labels.reshape(-1,1,1,1*k)[batch_inds]
    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # bbox_targets = bbox_transform(anchors, gt_boxes[argmax_overlaps, :]
    pos_anchors=all_anchors[inds_inside[labels==1]]
    bbox_targets = bbox_transform(pos_anchors, gt_boxes[argmax_overlaps, :][labels==1])
    bbox_targets = unmap(bbox_targets, total_anchors, inds_inside[labels==1], fill=0)
    batch_bbox_targets = bbox_targets.reshape(-1,1,1,4*k)[batch_inds]
    padded_fcmap=np.pad(feature_map,((0,0),(1,1),(1,1),(0,0)),mode='constant')
    padded_fcmap=np.squeeze(padded_fcmap)
    batch_tiles=[]
    for ind in batch_inds:
        x = ind % width
        y = int(ind/width)
        fc_1x1=padded_fcmap[y:y+3,x:x+3,:]
        batch_tiles.append(fc_1x1)
    return np.asarray(batch_tiles), batch_label_targets.tolist(), batch_bbox_targets.tolist()

ILSVRC_dataset_path='/home/jk/faster_rcnn/'
img_path=ILSVRC_dataset_path+'Data/DET/train/'
anno_path=ILSVRC_dataset_path+'Annotations/DET/train/'
feature_map_path='/home/jk/faster_rcnn/feature_maps/'
import glob
from multiprocessing import Process, Queue

BATCH_SIZE=256
def worker(path, q):
    print('worker start')
    batch_tiles=[]
    batch_labels=[]
    batch_bboxes=[]
    #'/ImageSets/DET/train_*'
    for fname in glob.glob(ILSVRC_dataset_path+path):
        print(fname)
        with open(fname,'r') as f:
            for line in f:
                if 'extra' not in line:
                    feature_map_file=feature_map_path+line.split()[0]
                    if not os.path.exists(feature_map_file):
                        continue

                    try:
                        category, gt_boxes, h_w = parse_label(anno_path+line.split()[0]+'.xml')
                        if len(gt_boxes)==0:
                            continue
                        tiles, labels, bboxes = produce_batch(feature_map_file, gt_boxes, h_w)
                    except Exception:
                        print('parse label or produce batch failed: for: '+line.split()[0])
                        traceback.print_exc()
                        continue
                    for i in range(len(tiles)):
                        batch_tiles.append(tiles[i])
                        batch_labels.append(labels[i])
                        batch_bboxes.append(bboxes[i])
                        if(len(batch_tiles)==BATCH_SIZE):
                            a=np.asarray(batch_tiles)
                            b=np.asarray(batch_labels)
                            c=np.asarray(batch_bboxes)
                            if not a.any() or not b.any() or not c.any():
                                print("empty array found.")
                            q.put([a,b,c])
                            batch_tiles=[]
                            batch_labels=[]
                            batch_bboxes=[]

q = Queue(20)

p1 = Process(target=worker, args=('/ImageSets/DET/train_*[0-1].txt',q))
p1.start()
p2 = Process(target=worker, args=('/ImageSets/DET/train_*[2-3].txt',q))
p2.start()
p3 = Process(target=worker, args=('/ImageSets/DET/train_*[4-6].txt',q))
p3.start()
p4 = Process(target=worker, args=('/ImageSets/DET/train_*[7-9].txt',q))
p4.start()

def input_generator():
    count=0
    while 1:
        batch = q.get()
        yield batch[0], [batch[1], batch[2]]

# model.load_weights('./weights.hdf5')
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='./weights_3x3_3.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(input_generator(), steps_per_epoch=3000, epochs=100, callbacks=[checkpointer])