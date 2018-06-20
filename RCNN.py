import os
import traceback
import numpy as np
import numpy.random as npr
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from keras.layers import Conv2D, TimeDistributed, Flatten, Dense, BatchNormalization
from keras.models import Input, Model, Layer
from keras.applications import InceptionResNetV2
from keras.preprocessing.image import load_img, img_to_array
from utils import generate_anchors, draw_anchors, bbox_overlaps, bbox_transform,\
                    loss_cls, smoothL1, parse_label, unmap, filter_boxes, \
                    clip_boxes, py_cpu_nms, bbox_transform_inv


##################  R-CNN Model  #######################
# RoI Pooling layer
class RoIPooling(Layer):
    def __init__(self, size=(7, 7)):
        self.size = size
        super(RoIPooling, self).__init__()

    def build(self, input_shape):
        self.shape = input_shape
        super(RoIPooling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        ind=K.reshape(inputs[2],(-1,))
        x = K.tf.image.crop_and_resize(inputs[0], inputs[1], ind, self.size)
        return x

    def compute_output_shape(self, input_shape):
        a=input_shape[1][0]
        b=self.size[0]
        c=self.size[1]
        d=input_shape[0][3]
        return (a,b,c,d)


BATCH=256

feature_map=Input(batch_shape=(None,None,None,1536))
rois=Input(batch_shape=(None, 4))
ind=Input(batch_shape=(None, 1),dtype='int32')

p1=RoIPooling()([feature_map, rois, ind])

flat1 = Flatten()(p1)


fc1 = Dense(
        units=1024,
        activation="relu",
        name="fc2"
    )(flat1)
fc1=BatchNormalization()(fc1)
output_deltas = Dense(
        units=4 * 200,
        activation="linear",
        kernel_initializer="uniform",
        name="deltas2"
    )(fc1)

output_scores = Dense(
        units=1 * 200,
        activation="softmax",
        kernel_initializer="uniform",
        name="scores2"
    )(fc1)

model=Model(inputs=[feature_map, rois, ind],outputs=[output_scores,output_deltas])
model.summary()
model.compile(optimizer='rmsprop',
            loss={'deltas2':smoothL1, 'scores2':'categorical_crossentropy'})

##################  prepare batch  #######################

FG_FRAC=.25
FG_THRESH=.5
BG_THRESH_HI=.5
BG_THRESH_LO=.1

#load an example to void graph problem
#TODO fix this.
pretrained_model = InceptionResNetV2(include_top=False)
img=load_img("./ILSVRC2014_train_00010391.JPEG")
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
not_used=pretrained_model.predict(x)

rpn_model = load_model('weights.hdf5',
            custom_objects={'loss_cls': loss_cls,'smoothL1':smoothL1})
not_used=rpn_model.predict(np.load('n02676566_6914')['fc'])

def produce_batch(filepath, gt_boxes, h_w, category):
    img=load_img(filepath)
    img_width=np.shape(img)[1] * scale[1]
    img_height=np.shape(img)[0] * scale[0]
    img=img.resize((int(img_width),int(img_height)))
    #feed image to pretrained model and get feature map
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    feature_map=pretrained_model.predict(img)
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
    # feed feature map to pretrained RPN model, get proposal labels and bboxes.
    res=rpn_model.predict(feature_map)
    scores=res[0]
    scores=scores.reshape(-1,1)
    deltas=res[1]
    deltas=np.reshape(deltas,(-1,4))
    # proposals transform to bbox values (x1, y1, x2, y2)
    proposals =bbox_transform_inv(all_anchors, deltas)
    proposals = clip_boxes(proposals, (h_w[0],h_w[1]))
    # remove small boxes, here threshold is 40 pixel
    keep = filter_boxes(proposals, 40)
    proposals = proposals[keep, :]
    scores = scores[keep]

    # sort socres and only keep top 6000.
    pre_nms_topN=6000
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]
    # apply NMS to to 6000, and then keep top 300
    post_nms_topN=300
    keep = py_cpu_nms(np.hstack((proposals, scores)), 0.7)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]
    # add gt_boxes to proposals.
    proposals=np.vstack( (proposals, gt_boxes) )
    # calculate overlaps of proposal and gt_boxes
    overlaps = bbox_overlaps(proposals, gt_boxes)
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    # labels = gt_labels[gt_assignment] #?

    # sub sample
    fg_inds = np.where(max_overlaps >= FG_THRESH)[0]
    fg_rois_per_this_image = min(int(BATCH*FG_FRAC), fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
    bg_inds = np.where((max_overlaps < BG_THRESH_HI) &
                       (max_overlaps >= BG_THRESH_LO))[0]
    bg_rois_per_this_image = BATCH - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    # labels = labels[keep_inds]
    rois = proposals[keep_inds]
    gt_rois=gt_boxes[gt_assignment[keep_inds]]
    targets = bbox_transform(rois, gt_rois)#input rois
    rois_num=targets.shape[0]
    batch_box=np.zeros((rois_num, 200, 4))
    for i in range(rois_num):
        batch_box[i, category] = targets[i]
    batch_box = np.reshape(batch_box, (rois_num, -1))
    # get gt category
    batch_categories = np.zeros((rois_num, 200, 1))
    for i in range(rois_num):
        batch_categories[i, category] = 1
    batch_categories = np.reshape(batch_categories, (rois_num, -1))
    return rois, batch_box, batch_categories

##################  generate data  #######################
ILSVRC_dataset_path='/home/jk/faster_rcnn/'
img_path=ILSVRC_dataset_path+'Data/DET/train/'
anno_path=ILSVRC_dataset_path+'Annotations/DET/train/'
import glob
from multiprocessing import Process, Queue

def worker(path):
    print('worker start ' + path)
    batch_rois=[]
    batch_featuremap_inds=[]
    batch_categories=[]
    batch_bboxes=[]
    fc_index=0
    dataset={}
    #'/ImageSets/DET/train_*'
    for fname in glob.glob(ILSVRC_dataset_path+path):
        print(fname)
        with open(fname,'r') as f:
            basename = os.path.basename(fname)
            category = int(basename.split('_')[1].split('.')[0])
            content=[]
            for line in f:
                if 'extra' not in line:
                    content.append(line)
            dataset[category]=content
    print(len(dataset))
    from random import randint
    while 1:
        try:
            category = randint(1, 200)
            content=dataset[category]
            n=randint(0,len(content))
            line=content[n]
            _, gt_boxes, h_w = parse_label(anno_path+line.split()[0]+'.xml')
            if len(gt_boxes)==0:
                continue
            rois, bboxes, categories = produce_batch(img_path+line.split()[0]+'.JPEG', gt_boxes, h_w, category)
        except Exception:
            # print('parse label or produce batch failed: for: '+line.split()[0])
            # traceback.print_exc()
            continue
        if len(rois) <= 0 :
            continue

        for i in range(len(rois)):
            batch_rois.append(rois[i])
            batch_featuremap_inds.append(fc_index)
            batch_categories.append(categories[i])
            batch_bboxes.append(bboxes[i])
        a=feature_map
        b=np.asarray(batch_rois)
        c=np.asarray(batch_featuremap_inds)
        d=np.asarray(batch_categories)
        e=np.asarray(batch_bboxes)
        f=np.zeros((len(rois),a.shape[1],a.shape[2],a.shape[3]))
        f[0]=feature_map[0]
        yield [f,b,c], [d,e]
        batch_rois=[]
        batch_featuremap_inds=[]
        batch_categories=[]
        batch_bboxes=[]
        fc_index=0

##################   start train   #######################
# model.load_weights('./rcnn_weights_1.hdf5')
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='./rcnn_weights_2.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(worker('/ImageSets/DET/train_*.txt'), steps_per_epoch=1000, epochs=100, callbacks=[checkpointer])