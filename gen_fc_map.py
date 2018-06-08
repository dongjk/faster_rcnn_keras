import os
import traceback
import pickle
import numpy as np
from keras.applications import InceptionResNetV2
from keras.preprocessing.image import load_img, img_to_array
from utils import parse_label
ILSVRC_dataset_path='/home/jk/wi/ILSVRC/'
img_path=ILSVRC_dataset_path+'Data/DET/train/'
anno_path=ILSVRC_dataset_path+'/Annotations/DET/train/'
import glob

pretrained_model = InceptionResNetV2(include_top=False)

for fname in glob.glob(ILSVRC_dataset_path+'/ImageSets/DET/train_*'):
    with open(fname,'r') as f:
        print(fname)
        for line in f:
            if 'extra' not in line:
                if os.path.exists('/home/jk/faster_rcnn/feature_maps/'+line.split()[0]):
                    continue
                try:
                    category, gt_boxes, scale = parse_label(anno_path+line.split()[0]+'.xml')
                    if len(gt_boxes)==0:
                        continue
                    img=load_img(img_path+line.split()[0]+'.JPEG')
                    img_width=np.shape(img)[1] * scale[1]
                    img_height=np.shape(img)[0] * scale[0]
                    img=img.resize((int(img_width),int(img_height)))
                    #feed image to pretrained model and get feature map
                    img = img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    feature_map=pretrained_model.predict(img)
                    output_filename='/home/jk/faster_rcnn/feature_maps/'+line.split()[0]
                    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                    fileObject = open(output_filename,'wb')
                    np.savez_compressed(fileObject,fc=feature_map)

                except Exception:
                    print('parse label or produce batch failed: for: '+line.split()[0])
                    traceback.print_exc()
                    continue