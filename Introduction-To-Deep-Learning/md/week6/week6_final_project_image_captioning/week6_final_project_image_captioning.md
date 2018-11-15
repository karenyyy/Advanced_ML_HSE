
# Image Captioning Final Project



Model architecture: CNN encoder and RNN decoder. 
(https://research.googleblog.com/2014/11/a-picture-is-worth-thousand-coherent.html)

# Import stuff


```python
import tensorflow as tf
from tensorflow.contrib import keras
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
L = keras.layers
K = keras.backend
import tqdm
import utils
import time
import zipfile
import json
from collections import defaultdict
import re
import random
from random import choice
import os
```

# Download data

Takes 10 hours and 20 GB. We've downloaded necessary files for you.

Relevant links (just in case):
- train images http://msvocds.blob.core.windows.net/coco2014/train2014.zip
- validation images http://msvocds.blob.core.windows.net/coco2014/val2014.zip
- captions for both train and validation http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip

# Extract image features

We will use pre-trained InceptionV3 model for CNN encoder (https://research.googleblog.com/2016/03/train-your-own-image-classifier-with.html) and extract its last hidden layer as an embedding:



```python
IMG_SIZE = 299
```

#### InceptionV3


```python
# we take the last hidden layer of InceptionV3 as an image embedding
def get_cnn_encoder():
    K.set_learning_phase(False)
    model = keras.applications.InceptionV3(include_top=False)
    preprocess_for_model = keras.applications.inception_v3.preprocess_input
    model = keras.models.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output))
    return model, preprocess_for_model
```

#### ResNet50


```python
# we take the last hidden layer of ResNet50 as an image embedding
def get_cnn_encoder():
    K.set_learning_phase(False)
    model = keras.applications.ResNet50(include_top=False)
    preprocess_for_model = keras.applications.resnet50.preprocess_input
    model = keras.models.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output))
    return model, preprocess_for_model
```


```python
K.clear_session()
encoder, preprocess_for_model = get_cnn_encoder()
```


```python
encoder
```




    <tensorflow.python.keras.engine.training.Model at 0x7f62b385cd30>




```python
preprocess_for_model
```




    <function tensorflow.python.keras.applications.inception_v3.preprocess_input(x)>



### data processing pipeline


```python
import os
import queue
import threading
import zipfile
import tqdm
import cv2
import numpy as np
import pickle
```


```python
def image_center_crop(img):
    h, w = img.shape[0], img.shape[1]
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    if h > w:
        diff = h - w
        pad_top = diff - diff // 2
        pad_bottom = diff // 2
    else:
        diff = w - h
        pad_left = diff - diff // 2
        pad_right = diff // 2
    return img[pad_top:h-pad_bottom, pad_left:w-pad_right, :]
```


```python
def decode_image_from_buf(buf):
    img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
```


```python
def crop_and_preprocess(img, input_shape, preprocess_for_model):
    img = image_center_crop(img)  # take center crop
    img = cv2.resize(img, input_shape)  # resize for our model
    img = img.astype("float32")  # prepare for normalization
    img = preprocess_for_model(img)  # preprocess for model
    return img
```


```python
def apply_model(zip_fn, model, preprocess_for_model, extensions=(".jpg",), input_shape=(224, 224), batch_size=32):
    # queue for cropped images
    q = queue.Queue(maxsize=batch_size * 10)

    # when read thread put all images in queue
    read_thread_completed = threading.Event()

    # time for read thread to die
    kill_read_thread = threading.Event()

    def reading_thread(zip_fn):
        zf = zipfile.ZipFile(zip_fn)
        for fn in tqdm.tqdm_notebook(zf.namelist()):
            if kill_read_thread.is_set():
                break
            if os.path.splitext(fn)[-1] in extensions:
                buf = zf.read(fn)  # read raw bytes from zip for fn
                img = decode_image_from_buf(buf)  # decode raw bytes
                img = crop_and_preprocess(img, input_shape, preprocess_for_model)
                while True:
                    try:
                        q.put((os.path.split(fn)[-1], img), timeout=1)  # put in queue
                    except queue.Full:
                        if kill_read_thread.is_set():
                            break
                        continue
                    break

        read_thread_completed.set()  # read all images

    # start reading thread
    t = threading.Thread(target=reading_thread, args=(zip_fn,))
    t.daemon = True
    t.start()

    img_fns = []
    img_embeddings = []

    batch_imgs = []

    def process_batch(batch_imgs):
        print(len(batch_imgs), len(batch_imgs[0]))
        batch_imgs = np.stack(batch_imgs, axis=0)
        print(len(batch_imgs), len(batch_imgs[0]))
        # shape of batch_imgs (samples, 299, 299) -> (samples, 2048)
        batch_embeddings = model.predict(batch_imgs)
        print(len(batch_embeddings), len(batch_embeddings[0]))
        img_embeddings.append(batch_embeddings)

    try:
        while True:
            try:
                fn, img = q.get(timeout=1)
            except queue.Empty:
                if read_thread_completed.is_set():
                    break
                continue
            img_fns.append(fn)
            batch_imgs.append(img)
            if len(batch_imgs) == batch_size:
                process_batch(batch_imgs)
                # flush the cache and start over
                batch_imgs = []
            q.task_done()
        # process last batch
        if len(batch_imgs):
            process_batch(batch_imgs)
    finally:
        kill_read_thread.set()
        t.join()

    q.join()

    img_embeddings = np.vstack(img_embeddings)
    return img_embeddings, img_fns
```


```python
train_img_embeds, train_img_fns = apply_model(
    "/home/karen/Downloads/data/train2014.zip", encoder, preprocess_for_model, input_shape=(IMG_SIZE, IMG_SIZE))
```


    HBox(children=(IntProgress(value=0, max=82784), HTML(value='')))


    32 299
    32 299
    32 2048
    32 299
    32 299



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-49-dd745125d295> in <module>()
          1 train_img_embeds, train_img_fns = apply_model(
    ----> 2     "/home/karen/Downloads/data/train2014.zip", encoder, preprocess_for_model, input_shape=(IMG_SIZE, IMG_SIZE))
    

    <ipython-input-48-8fdc4b0941e8> in apply_model(zip_fn, model, preprocess_for_model, extensions, input_shape, batch_size)
         58             batch_imgs.append(img)
         59             if len(batch_imgs) == batch_size:
    ---> 60                 process_batch(batch_imgs)
         61                 # flush the cache and start over
         62                 batch_imgs = []


    <ipython-input-48-8fdc4b0941e8> in process_batch(batch_imgs)
         43         batch_imgs = np.stack(batch_imgs, axis=0)
         44         print(len(batch_imgs), len(batch_imgs[0]))
    ---> 45         batch_embeddings = model.predict(batch_imgs)
         46         print(len(batch_embeddings), len(batch_embeddings[0]))
         47         img_embeddings.append(batch_embeddings)


    /usr/local/lib/python3.5/dist-packages/tensorflow/python/keras/engine/training.py in predict(self, x, batch_size, verbose, steps)
       1491     else:
       1492       return training_arrays.predict_loop(
    -> 1493           self, x, batch_size=batch_size, verbose=verbose, steps=steps)
       1494 
       1495   def train_on_batch(self, x, y=None, sample_weight=None, class_weight=None):


    /usr/local/lib/python3.5/dist-packages/tensorflow/python/keras/engine/training_arrays.py in predict_loop(model, inputs, batch_size, verbose, steps)
        372         ins_batch[i] = ins_batch[i].toarray()
        373 
    --> 374       batch_outs = f(ins_batch)
        375       if not isinstance(batch_outs, list):
        376         batch_outs = [batch_outs]


    /usr/local/lib/python3.5/dist-packages/tensorflow/python/keras/backend.py in __call__(self, inputs)
       2912       self._make_callable(feed_arrays, feed_symbols, symbol_vals, session)
       2913 
    -> 2914     fetched = self._callable_fn(*array_vals)
       2915     self._call_fetch_callbacks(fetched[-len(self._fetches):])
       2916     return fetched[:len(self.outputs)]


    /usr/local/lib/python3.5/dist-packages/tensorflow/python/client/session.py in __call__(self, *args, **kwargs)
       1380           ret = tf_session.TF_SessionRunCallable(
       1381               self._session._session, self._handle, args, status,
    -> 1382               run_metadata_ptr)
       1383         if run_metadata:
       1384           proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)


    KeyboardInterrupt: 



```python
def save_pickle(obj, fn):
    with open(fn, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('pickle saved!')
```


```python
save_pickle(train_img_embeds, "train_img_embeds.pickle")
save_pickle(train_img_fns, "train_img_fns.pickle")
```


```python
val_img_embeds, val_img_fns = apply_model(
    "val2014.zip", encoder, preprocess_for_model, input_shape=(IMG_SIZE, IMG_SIZE))
save_pickle(val_img_embeds, "val_img_embeds.pickle")
save_pickle(val_img_fns, "val_img_fns.pickle")
```


```python
def read_pickle(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)
```


```python
# load prepared embeddings
train_img_embeds = read_pickle("/home/karen/Downloads/data/train_img_embeds.pickle")
train_img_fns = read_pickle("/home/karen/Downloads/data/train_img_fns.pickle")
val_img_embeds = read_pickle("/home/karen/Downloads/data/val_img_embeds.pickle")
val_img_fns = read_pickle("/home/karen/Downloads/data/val_img_fns.pickle")
# check shapes
print(train_img_embeds.shape, len(train_img_fns))
print(val_img_embeds.shape, len(val_img_fns))
```

    (82783, 2048) 82783
    (40504, 2048) 40504



```python
def sample_zip(fn_in, fn_out, rate=0.01, seed=42):
    np.random.seed(seed)
    with zipfile.ZipFile(fn_in) as fin, zipfile.ZipFile(fn_out, "w") as fout:
        sampled = filter(lambda _: np.random.rand() < rate, fin.filelist)
        for zInfo in sampled:
            fout.writestr(zInfo, fin.read(zInfo))

sample_zip("/home/karen/Downloads/data/train2014.zip", "/home/karen/Downloads/data/train2014_sample.zip")
sample_zip("/home/karen/Downloads/data/val2014.zip", "/home/karen/Downloads/data/val2014_sample.zip")
```


```python
# check prepared samples of images
list(filter(lambda x: x.endswith("_sample.zip"), os.listdir("/home/karen/Downloads/data/")))
```




    ['train2014_sample.zip', 'val2014_sample.zip']



# Extract captions for images


```python
# extract captions from zip
def get_captions_for_fns(fns, zip_fn, zip_json_path):
    zf = zipfile.ZipFile(zip_fn)
    j = json.loads(zf.read(zip_json_path).decode("utf8")) 
    id_to_fn = {img["id"]: img["file_name"] for img in j["images"]} 
    fn_to_caps = defaultdict(list)
    for cap in j['annotations']:
        fn_to_caps[id_to_fn[cap['image_id']]].append(cap['caption'])
    fn_to_caps = dict(fn_to_caps)
    return list(map(lambda x: fn_to_caps[x], fns))
    
train_captions = get_captions_for_fns(train_img_fns, "/home/karen/Downloads/data/captions_train-val2014.zip", 
                                      "annotations/captions_train2014.json")

val_captions = get_captions_for_fns(val_img_fns, "/home/karen/Downloads/data/captions_train-val2014.zip", 
                                      "annotations/captions_val2014.json")
# check shape
print(len(train_img_fns), len(train_captions))
print(len(val_img_fns), len(val_captions))
```

    82783 82783
    40504 40504



```python
zp = zipfile.ZipFile("/home/karen/Downloads/data/captions_train-val2014.zip")
j=json.loads(zp.read("annotations/captions_train2014.json").decode('utf-8'))
j['images'][0], j['annotations'][0]
```




    ({'coco_url': 'http://mscoco.org/images/57870',
      'date_captured': '2013-11-14 16:28:13',
      'file_name': 'COCO_train2014_000000057870.jpg',
      'flickr_url': 'http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg',
      'height': 480,
      'id': 57870,
      'license': 5,
      'width': 640},
     {'caption': 'A very clean and well decorated empty bathroom',
      'id': 48,
      'image_id': 318556})




```python
zf = zipfile.ZipFile("/home/karen/Downloads/data/train2014_sample.zip")
zf.filelist[0].filename.rsplit('/')
```




    ['train2014', 'COCO_train2014_000000092353.jpg']




```python
all_files = set(train_img_fns)
l = list(filter(lambda x: x.filename.rsplit("/")[-1] in all_files, zf.filelist))
zf.read(l[0])
```




    b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xe1\x0b\x0eXMP\x00://ns.adobe.com/xap/1.0/\x00<?xpacket begin=\'\xef\xbb\xbf\' id=\'W5M0MpCehiHzreSzNTczkc9d\'?>\n<x:xmpmeta xmlns:x=\'adobe:ns:meta/\' x:xmptk=\'Image::ExifTool 6.54\'>\n<rdf:RDF xmlns:rdf=\'http://www.w3.org/1999/02/22-rdf-syntax-ns#\'>\n\n <rdf:Description rdf:about=\'\'\n  xmlns:xmp=\'http://ns.adobe.com/xap/1.0/\'>\n  <xmp:CreatorTool>picnik.com</xmp:CreatorTool>\n </rdf:Description>\n</rdf:RDF>\n</x:xmpmeta>\n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n                                                                                                    \n<?xpacket end=\'w\'?>\xff\xdb\x00C\x00\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x02\x02\x03\x02\x02\x02\x02\x02\x04\x03\x03\x02\x03\x05\x04\x05\x05\x05\x04\x04\x04\x05\x06\x07\x06\x05\x05\x07\x06\x04\x04\x06\t\x06\x07\x08\x08\x08\x08\x08\x05\x06\t\n\t\x08\n\x07\x08\x08\x08\xff\xdb\x00C\x01\x01\x01\x01\x02\x02\x02\x04\x02\x02\x04\x08\x05\x04\x05\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\xff\xc0\x00\x11\x08\x01\xf4\x01\xd8\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x04\x03\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x05\x02\x03\x04\x06\x00\x01\x07\x08\t\n\x0b\xff\xc4\x00C\x10\x00\x02\x01\x03\x03\x02\x04\x03\x06\x03\x07\x04\x02\x02\x01\x05\x01\x02\x03\x04\x05\x11\x06\x12!\x001\x07\x13"A\x14Qa\x08\x15#q\x81\x912\xa1\xf0\t\x16B\xb1\xc1\xd1\xe1$3R\xf1\n\x17br%4\x18S\x82C\xff\xc4\x00\x1d\x01\x00\x02\x03\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x03\x04\x01\x02\x05\x06\x00\x07\x08\t\xff\xc4\x00J\x11\x00\x01\x03\x03\x03\x02\x04\x02\x07\x06\x03\x07\x03\x03\x03\x05\x01\x02\x03\x11\x00\x04!\x05\x121AQ\x06\x13"aq\x81\x07\x142\x91\xa1\xb1\xf0\x08\x15#B\xc1\xd1Rb\xe1\x16$3r\x82\x92\xf1\x17\xa2\xb2\x18C\xd24cs\xc2S\x83\x93\xa3\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xfd>\x9a\x06ERay;d\x85\x0b\xees\xc6x\xf6\xf6\xeb\xf6bn\xab\xf9\x92\xdb.U\x8a\x8a\x968J\x92\x17\xcc\xe1s\xdb\x1f\x9e;\xfe|\xf7\xeb2\xe1\xea\xd5\xb6\x18\xcd\x1c\x82\x88\xb1<2\x92\x08\xf5\x00O\xb7\x1f\x97\x07="\xeb\xe2\xb4\x19\xb0\x9cT\xb8\xac\x15\x1b\x97b\x94\r\x8d\xc8\x18\x8c~\x9d\xf1\xd0N\xa2\x90+U\x9d*b\x8f\xd3i\xb2\x10\xb1f2e\xbf\xed.0OoQ\xf9`q\xd6s\xba\xa4\x98\xad\x84\xe8\xb8"\x9d\x8a\xc0\xa1\xb1*\x12I\xe0g\x19\x1f\xe9\xee:\xa9\xd4\xccb\xac\xc6\x97\xb4\x81\xde\x8dSPR\xc0\xb1\xa2\x86\x89B\xe0n\xc1\xfa\x0c\x10?\xaf\xcb= \xed\xc2\xd7\x93[\r\xd9\xf9|Q\x8aZe\x01\x04d6\x00Q\x93\xdf\x03\xb7\x1e\xfe\xfe\xfd"\xe3\xb5\xa5l\xd4(M\x19\x8a\x96\x19\x9c\x96v\x0e\x00\n\t \x8fo\xcb\xdc\xfd:Io\x10kY<T\xa3N\xe1\xcb`H\xbd\x88,\x00\xfd\xfd\xf1\xdf\xa1\x17\xa4W\x92\xdeiiO\xe6?,\xeaI\xe3\x90p~C\xfc\xfe}\rN\x94\x8a\xb2\x98\xc5-\xad\x9eh\x04\x95\xf6\xceT\x12\xdf\x90\x1f\xafP\x9b\xb83V\x16\xb0\x99\xa9\t\xa5\x92b\x1cI\x1b\xe3>\xa0p\x7fr\x07\x1f\xef\xd0U\xad\x14\x98\x8aY\xcd((\xca\xaaBh\x88H\xdd#\xec\r\x8cz\xb2s\xd5\x0e\xbe\xae\x94\xcbZ*\x08\xc5m\xb4e>\xd6X\x19\xf7s\x9fW#\xeb\x83\xf5 \xf5\x1f\xbf\x95\xd6\xac\xad>\x04T\x05\xd3\xb2R\xe4\x80\xb3\x81\x900N>\xa4\xfc\xfbd~_^\x8cu=\xe2\xa1:l\xd18\xac\xf9\x93\xcbg\xf2"\x07nU\xb9_\x9e\x0f\xb0\xed\xd2\xeb\xbe\xebGN\x93\xde\x8cSi\xe8\x98\x80\'\x9b\xcc\x03 \xa9\xc8\xc61\xd8\xf4\x93\x9a\x84\x1ay\xab^\x94\xec\x9aN\x9ax\xca,\xb27r\x030<g\xbf\xe5\xd5\x13\xac)9\xa6Sa\xbb\x15\xa1\xa5>\x191N?\x18\x1fIn9\x1e\xc7\x9eA\xeb\xca\xd6\x02\xfe\xd7\x15G\xb4\xb8\xe2\x98:f\xb9\xfc\xa1 \x8d\x9b\xff\x005\xf9\xfc\x8f\xd3\xfd\xba\xb2uT\x8f\xb3J\xabE\xdcA\xedF\xa8\xf4\xbc\r\x8f\x8a\xa7\x8ec\x83\x96a\xeaRF\x0e\x0f\xbfl~\xdd$\xf6\xaa\xa1\xf6MmZ\xe9X\xdax\xa2\xf1iKx\xda\xd1C\xe5\x91\xdf\xd4\x7f\xd7\xf3<\xf4\xa7\xefw84\xd9\xf0\xe3J\xf5&\xa4&\x9b\xa0l\xb1\x8bq\x07#88\xfe\xb9\xe8_\xbc\xdc4V\xb4\x16\xc55S\xa5hg\x00J\xccT\x0e\xc3\x03\x1f\xeb\xed\xfa\xf5tjK\x19\xaa\xbd\xe1\xf6\xd5P\x93HPB\xe6d\xf3\t\xfa\x9f\x7f\x9f\xe7\xf5\xe9\x95k.(A\xa4U\xe1\xf0\x9f\xb3S`\xb4\xfc<\xe1\x96fd\xeeG\x1c\xf4\xaa\xef\t\x10hv\xfa6\xc5\xd5\x8a*oA\xf9\x93\xfbu\x9aU5\xd5"\xd7\xd3H\x96\x91\x01>\x81\xf3\xfc\xff\x00>\xae\x87\x080*\xaeZ\x88\xcd\x0bkl\x1eg\x9a"\x8f~;\x91\x9e\x98\xfa\xc185\x92\xfd\x8c\xe6\xa2KD\t*#\x00\x02\x0f\x07\x1c\x8e\x8c\xdb\xd0"\xb2\xdd\xb2\x81\xc50\x94\xfeP\x03\xd21\x8fn\xc7\xe7\xff\x00\x1d\x189\xbb4\x14\xda\xe2\xa2URy\xc0\xe0!_|\x80\x0f\x7fa\xd1\x9az)+\xbbi\xc5CJ\x06.B\xab\x11\xc7\xf1q\xfdv\xe8\xeb\xb91Y\xadi\xf9\x8a-\x05\x00\x0e\xcc\x15U\x89\xe4\x00\x08\xff\x00\x9e\xdd$\xe5\xd4\x88\xad\x96,\xc8\xc5\x11\x10\x10\t\xc9\x07\xe7\x81\x9e\x80\x1c\x15\xa2\x191L\xb59\xdeN\xef\x7f\xf0\xf1\xd1\x03\x98\xa5\x85\xa6\xd5\x95V<\x00.\xd6\x0er0}\xf3\xf9\xf5\xe0\xe9\xa9v\xdaD\x9aR\xa3\x85\xceA\xfa\xfc\x87T&\xa1-\xe2\x90\xf2\x00\x0ew\xe3\xb0\xe7\xbf\xe5\xd7\xa8\xb1B\xe7\xf2\xdd\xb0\xc0\x11\xfcM\x83\x8f\xdf\xf6=8\xd1\x8a\xc8\xb8\xebC\xfc\xc8\xd88b\x0b\xb0?\xe1\xed\xfd\x7f\xa0\xe9\x88\xedX\xf8\xa1\xed4J\xaeIE#\x18\xe38\xe7\x8f\xe7\x8e\x9a\rVc\xc2\x85\xd5E\x13\x96f\x96\x1d\xb9,\xf8\x00\x85\xf9\x9e\x9ci\xdfjY\xcf]\x03\x9e\xdb\x17\xa5\xb7\x9c\x8ep01\x8f\x97\x1f\xaezy\xab\x93K\x0bM\xf42kb\xb0\xda\x14n=\xd8`\x01\xc9\xe0\xf1\xdf\x9e\x9aE\xd7Z\xa5\xc5\x86(}E\x8d\xa4-\x99\x9d\x90\x80p\x18\x9f\xd3\x93\x83\xfe\xbd2\x8b\xf8\x15\xcf\xaa\xc2\x08\xa6E\x9a3\xbd\x8a\x80\xac\t\n\xe0\xa8\xc7\xcb\x9f\xa6;}:\xbf\xef\x0e\x82\x88\xab^\xf4\xfaZ\xa8\xa3WGD,A\x03\ny\xe7\x1d\xbb\x0e?\xcf\xaa\xaa\xe5d\xee\x15f\xadc4\xcbZ\xad\xe0\x03\x0cJ\n\x9e1\x81\xb7\xfep\x0f\xf5\xcfW7N\x7f1\xa4\xee\x98\x9cT)m\x02I\x11\xd0\x8d\xa0\x81\x827\x11\xcf\x07\x8f\xd3\xa3\xa6\xee\x04V\x1b\xdaF\xf14B\x92\xd2ah\xc9V\xde\xa0\x0c\x7f\xe3\xfc=\xfe\xb9\x18\xe97\xaf\'\x15\xa9c\xa6\xec\x11EZ\xc8&\x01\xc0U\x93>\x91\x8c\x03\xf5>\xfe\xe7\xe7\x82zL^\x94\x98\xad1\xa4\xff\x005\x11M<\xa4\xa9\xf2\xb7\xa8\xe7\x03\x03q\xf9\xf4\x175\x03\x15\xaa\xc5\x8c\t\xa2Q\xd8\x16lb(\x98\x13\xfc@dq\xd8\x81\xfe\xfd(\xbdKh\x9a\xd1F\x97\xbcM<\xdaj\x18\x11\x8aG"\xb9\x07\x80;g\xf2\xff\x00\x9e\xa8\x9d^Nk\xcehr"\x80\xd6iu\x91\x0e\xdd\xc0\x0fW\x07\xbf\xe7\xc0\xc7Z\x0cj\xe4VE\xd6\x88@\x94\xd0g\xd2\x0c\xc5\xa2\x8a,\x0eN\xe0}\xff\x00\x7f\xaf\xf2\xebC\xf7\xd7SX\xcb\xf0\xe2Nz\xd2\x1bE\xc6\xf1\xb2\xc3#\xc4\x0fl\xa99 \xf7\x1f\xd7\xb7R\x9dv\x0c\x915\x9e\xef\x86I\xc2k\xcb\x16\xcb\xe5\xba`\xac\xf3\xcd\x10R@R2\x0f\x07\xb9\xc7\xff\x00\xb7_[\xb9\xd3\xdcEp\x96\x9a\x93NU\xae\x8e\xba\x91\xcc\x7f\x89\x95R\x0e\x14\r\xd9=\xb9\xeb-\xf6\x96+u\x9b\xb9\xab\x05\x1dV6\xc6\xd3\xc6\xa39\xcf\x1c\x0e{\x8f\xa7?\xcf\xac\xa7\x9a#1[\xd6\xcf\n\xb6\xdb\xee\xe1\x0cqM\x14\x15\x08\xd8]\xc0go\xbf\xf0\xfb\xe0\x9e\xe3\xe5\xd6-\xd5\x9e\xe95\xd0\xe9\xf7\x88\x89\xab\x95%]+\xaa\xb2F\x94\xe8F\xf08$\x1c{\x8e1\xc6\x7f/\xd3\xacGY_\x15\xd6\xda^6\xb14b8\xa9jPH}\x12\x10O\xf0\x82\xdd\xb3\x908\xc7\xf5\xdb\xa4V\xe2\xd0b\xb5\xdbm\xb5\x8a}m\x03r\xb2\xaf\x9d\xbb\xdd\x0f\x19\xf9\xe7\xdb\xdf\x91\xdb\x8e\x86n\xa2f\x8a-Px\xa9B\xd8\x9bs\x80\xa4\x80\x07>\xd9\xec\x08\xfc\xc7\xfe\xba\x19\xba5\xe4\xd8\x00f\x92\xf6q"9r\x81\xc9\xec088>\xdd\xff\x00\xd3\xbf^E\xe4\x1a!\xb6\x02\x86O\x0c\xd0\xfaRY=D\xe4s\xe9<\xf2\x0f\xccq\xd3\xad\xa9*\xe6\x91y\xb3\x98\xe6\xa0\xadMb>Y\xce\xc2F\n\x8eTg\xe6\x7f\xae;\xf4o!\x07\x03\x9a]\x1b\xc1\x93D"\xae\xa8T\x08\x18\xa3n\xdb\x80\xc3\x8f\x91\x07\x8eG\xbf?.\x96U\xb0&i\xd0\xe9\x02h\xa4w6\x01\x8a\xb1v\x00\xb0\'\xb7lc\xf9\xf4\xa2\xad$\xc5QO\xee\xc5L\x8e\xfa\xf0\x85\x8dIC\x9eI\x07\xe5\xf3=\xbd\x87\xe9\xef\xd2\xc6\xc0\x15Q\xda\xbe\x81\xb6\xa7\x8dG\x08H\xcdB\x93\'\x04\xe0\xe7\x1c\xf7\xed\xf4\xcez\n\xb4\xc3>\x9a2u\x04\x8c\x9aZ_\xe8\x1c\x82\xb2\xb0\xe7\x1bW\xb9\xcfs\xcf\x18\xe3\xaa\x9d5\xca;z\x93d\xd1X.\x143d\x86X\xdb\x07\'!q\xf4\xcf\xeb\xd2\x8ba\xc4\xe2\x9c\xfa\xebj\x14z&\x8fh1>\xec\x1c\x9c7\x7f\xcf\xfa\xf6\xe9\x15\x839\xa7Y}?\xcbO\xc1"\x




<b>limit_output extension: Maximum message size of 10000 exceeded with 470150 characters</b>



```python
# look at training example (each has 5 captions)
def show_training_example(train_img_fns, train_captions, example_idx=0):
    """
    You can change example_idx and see different images
    """
    zf = zipfile.ZipFile("/home/karen/Downloads/data/train2014_sample.zip")
    captions_by_file = dict(zip(train_img_fns, train_captions))
    all_files = set(train_img_fns)
    # get sample image filenames
    found_files = list(filter(lambda x: x.filename.rsplit("/")[-1] in all_files, zf.filelist))
    example = found_files[example_idx]
    # zf.read() outputs binary code of images
    # need to decode from bin code to image arrays
    img = utils.decode_image_from_buf(zf.read(example))
    plt.imshow(utils.image_center_crop(img))
    plt.title("\n".join(captions_by_file[example.filename.rsplit("/")[-1]]))
    plt.show()
    
show_training_example(train_img_fns, train_captions, example_idx=142)
```


![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week6/output_32_0.png)


# Prepare captions for training


```python
# preview captions data
train_captions[:2], len(train_captions)
```




    ([['A long dirt road going through a forest.',
       'A SCENE OF WATER AND A PATH WAY',
       'A sandy path surrounded by trees leads to a beach.',
       'Ocean view through a dirt road surrounded by a forested area. ',
       'dirt path leading beneath barren trees to open plains'],
      ['A group of zebra standing next to each other.',
       'This is an image of of zebras drinking',
       'ZEBRAS AND BIRDS SHARING THE SAME WATERING HOLE',
       'Zebras that are bent over and drinking water together.',
       'a number of zebras drinking water near one another']],
     82783)




```python
# special tokens
PAD = "#PAD#"
UNK = "#UNK#"
START = "#START#"
END = "#END#"

from collections import Counter

# split sentence into tokens (split into lowercased words)
def split_sentence(sentence):
    return list(filter(lambda x: len(x) > 0, re.split('\W+', sentence.lower())))

def generate_vocabulary(train_captions):
    """
    Return {token: index} for all train tokens (words) that occur 5 times or more, 
        `index` should be from 0 to N, where N is a number of unique tokens in the resulting dictionary.
    Use `split_sentence` function to split sentence into tokens.
    Also, add PAD (for batch padding), UNK (unknown, out of vocabulary), 
        START (start of sentence) and END (end of sentence) tokens into the vocabulary.
    """
    sentences = [sentence for cap in train_captions for sentence in cap]
    sentences = split_sentence(' '.join(sentences))
    tokens = Counter(sentences)
    frequent_tokens = [key for key, value in tokens.items() if value>=5]
    vocab = frequent_tokens + [PAD, UNK, START, END]
    return {token: index for index, token in enumerate(sorted(vocab))}
    
def caption_tokens_to_indices(captions, vocab):
    """
    `captions` argument is an array of arrays:
    [
        [
            "image1 caption1",
            "image1 caption2",
            ...
        ],
        [
            "image2 caption1",
            "image2 caption2",
            ...
        ],
        ...
    ]
    Use `split_sentence` function to split sentence into tokens.
    Replace all tokens with vocabulary indices, use UNK for unknown words (out of vocabulary).
    Add START and END tokens to start and end of each sentence respectively.
    For the example above you should produce the following:
    [
        [
            [vocab[START], vocab["image1"], vocab["caption1"], vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"], vocab[END]],
            ...
        ],
        ...
    ]
    """
    final_result=[]
    img_result=[]
    for caption in captions:
        for sentence in caption:
            result=[vocab[START]]
            for token in split_sentence(sentence):
                if token in vocab: 
                    result+=[vocab[token]]
                else:
                    result+=[vocab[UNK]]
            result+=[vocab[END]]
            img_result.append(result)
            result=[]
        final_result.append(img_result)
        img_result=[]
                
    res = [[[vocab[START]] + [vocab[token] if token in vocab else vocab[UNK] for token in split_sentence(sentence)] + [vocab[END]] for sentence in caption] for caption in captions]
    return res
```


```python
# prepare vocabulary
vocab = generate_vocabulary(train_captions)
vocab_inverse = {idx: w for w, idx in vocab.items()}
print(len(vocab), vocab['#PAD#'])
```

    8769 1



```python
# replace tokens with indices
train_captions_indexed = caption_tokens_to_indices(train_captions, vocab)
val_captions_indexed = caption_tokens_to_indices(val_captions, vocab)
```


```python
len(train_captions_indexed), len(val_captions_indexed)
```




    (82783, 40504)



Captions have different length, but we need to batch them, that's why we will add PAD tokens so that all sentences have an euqal length. 

We will crunch LSTM through all the tokens, but we will ignore padding tokens during loss calculation.


```python
train_captions_indexed[0]
```




    [[2, 54, 4462, 2305, 6328, 3354, 7848, 54, 3107, 0],
     [2, 54, 6540, 5127, 8486, 249, 54, 5437, 8507, 0],
     [2, 54, 6502, 5437, 7581, 1124, 8052, 4287, 7905, 54, 639, 0],
     [2, 5120, 8367, 7848, 54, 2305, 6328, 7581, 1124, 54, 3108, 331, 0],
     [2, 2305, 5437, 4286, 710, 587, 8052, 7905, 5174, 5684, 0]]




```python
# get the longest encoded sentence
max(train_captions_indexed[0], key=lambda x:len(x))
```




    [2, 5120, 8367, 7848, 54, 2305, 6328, 7581, 1124, 54, 3108, 331, 0]




```python
# we will use this during training
def batch_captions_to_matrix(batch_captions, pad_idx, max_len=None):
    """
    `batch_captions` is an array of arrays:
    [
        [vocab[START], ..., vocab[END]],
        [vocab[START], ..., vocab[END]],
        ...
    ]
    Put vocabulary indexed captions into np.array of shape (len(batch_captions), columns),
        where "columns" is max(map(len, batch_captions)) when max_len is None
        and "columns" = min(max_len, max(map(len, batch_captions))) otherwise.
    Add padding with pad_idx where necessary.
    Input example: [[1, 2, 3], [4, 5]]
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=None
    Output example: np.array([[1, 2], [4, 5]]) if max_len=2
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=100
    Try to use numpy, we need this function to be fast!
    """
    matrix=[]
    batch_max = len(max(batch_captions, key=lambda x:len(x)))
    if max_len:
        max_len = min(max_len, batch_max)
    else:
        max_len = batch_max
    for caption in batch_captions:
        cap_len = len(caption)
        if cap_len<=max_len:
            caption+=[pad_idx]*(max_len-cap_len)
        else:
            caption = caption[:max_len]
        matrix.append(caption)
    matrix = np.array(matrix)
    return matrix
```


```python
batch_captions_to_matrix(train_captions_indexed[1], pad_idx=0)
```




    array([[   2,   54, 3484, 5127, 8755, 7296, 5025, 7905, 2540, 5222,    0],
           [   2, 7835, 3998,  242, 3877, 5127, 5127, 8756, 2468,    0,    0],
           [   2, 8756,  249,  764, 6709, 7804, 6484, 8492, 3738,    0,    0],
           [   2, 8756, 7800,  330,  711, 5254,  249, 2468, 8486, 7919,    0],
           [   2,   54, 5072, 5127, 8756, 2468, 8486, 4975, 5159,  269,    0]])




```python
# make sure you use correct argument in caption_tokens_to_indices
assert len(caption_tokens_to_indices(train_captions[:10], vocab)) == 10
assert len(caption_tokens_to_indices(train_captions[:5], vocab)) == 5
```

# Training

## Define architecture

Since our problem is to generate image captions, RNN text generator should be conditioned on image. The idea is to use image features as an initial state for RNN instead of zeros. 

Remember that you should transform image feature vector to RNN hidden state size by fully-connected layer and then pass it to RNN.

During training we will feed ground truth tokens into the lstm to get predictions of next tokens. 

Notice that we don't need to feed last token (END) as input



```python
train_img_embeds.shape
```




    (82783, 2048)




```python
IMG_EMBED_SIZE = train_img_embeds.shape[1] # 2048
IMG_EMBED_BOTTLENECK = 120
WORD_EMBED_SIZE = 100
LSTM_UNITS = 300
LOGIT_BOTTLENECK = 120
pad_idx = vocab[PAD]
```


```python
# remember to reset your graph if you want to start building it from scratch!
tf.reset_default_graph()
tf.set_random_seed(42)
s = tf.InteractiveSession()
```

    /usr/local/lib/python3.5/dist-packages/tensorflow/python/client/session.py:1645: UserWarning: An interactive session is already active. This can cause out-of-memory errors in some cases. You must explicitly call `InteractiveSession.close()` to release resources held by the other session(s).
      warnings.warn('An interactive session is already active. This can '



```python
class decoder:
    img_embeds = tf.placeholder('float32', [None, IMG_EMBED_SIZE]) # (None, 2048)
    sentences = tf.placeholder('int32', [None, None]) # batch_max_len might be different
    
    # we use bottleneck here to reduce the number of parameters
    # image embedding -> bottleneck
    img_embed_to_bottleneck = L.Dense(IMG_EMBED_BOTTLENECK, 
                                      input_shape=(None, IMG_EMBED_SIZE), 
                                      activation='elu')
    # image embedding bottleneck -> lstm initial state
    img_embed_bottleneck_to_h0 = L.Dense(LSTM_UNITS,
                                         input_shape=(None, IMG_EMBED_BOTTLENECK),
                                         activation='elu')
    # word -> embedding
    word_embed = L.Embedding(len(vocab), WORD_EMBED_SIZE)
    # lstm cell (from tensorflow)
    lstm = tf.nn.rnn_cell.LSTMCell(LSTM_UNITS)
    
    # we use bottleneck here to reduce model complexity
    # lstm output -> logits bottleneck
    token_logits_bottleneck = L.Dense(LOGIT_BOTTLENECK, 
                                      input_shape=(None, LSTM_UNITS),
                                      activation="elu")
    # logits bottleneck -> logits for next token prediction
    token_logits = L.Dense(len(vocab),
                           input_shape=(None, LOGIT_BOTTLENECK))
    
    # initial lstm cell state of shape (None, LSTM_UNITS),
    # we need to condition it on `img_embeds` placeholder.
    c0 = h0 = img_embed_bottleneck_to_h0(img_embed_to_bottleneck(img_embeds))

    # embed all tokens but the last for lstm input,
    # remember that L.Embedding is callable,
    # use `sentences` placeholder as input.
    word_embeds = word_embed(sentences[:, :-1]) # exclude the last char of sentence
    
    # during training we use ground truth tokens `word_embeds` as context for next token prediction.
    # that means that we know all the inputs for our lstm and can get 
    # all the hidden states with one tensorflow operation (tf.nn.dynamic_rnn).
    # `hidden_states` has a shape of [batch_size, time steps, LSTM_UNITS].
    hidden_states, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=word_embeds,
                                         initial_state=tf.nn.rnn_cell.LSTMStateTuple(c0, h0))

    # now we need to calculate token logits for all the hidden states
    
    # first, we reshape `hidden_states` to [-1, LSTM_UNITS]
    flat_hidden_states = tf.reshape(hidden_states, shape=[-1, LSTM_UNITS])
    
    # then, we calculate logits for next tokens using `token_logits_bottleneck` and `token_logits` layers
    flat_token_logits = token_logits(token_logits_bottleneck(flat_hidden_states))
    
    # then, we flatten the ground truth token ids.
    # remember, that we predict next tokens for each time step,
    # use `sentences` placeholder.
    flat_ground_truth = tf.reshape(sentences[:, 1:], [-1,])

    # we need to know where we have real tokens (not padding) in `flat_ground_truth`,
    # we don't want to propagate the loss for padded output tokens,
    # fill `flat_loss_mask` with 1.0 for real tokens (not pad_idx) and 0.0 otherwise.
    flat_loss_mask = tf.not_equal(flat_ground_truth, pad_idx)

    # compute cross-entropy between `flat_ground_truth` and `flat_token_logits` predicted by lstm
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=flat_ground_truth, 
        logits=flat_token_logits
    )

    # compute average `xent` over tokens with nonzero `flat_loss_mask`.
    # we don't want to account misclassification of PAD tokens, because that doesn't make sense,
    # we have PAD tokens for batching purposes only!
    loss = tf.reduce_mean(tf.boolean_mask(xent, flat_loss_mask))
```


```python
np.reshape([[1,2,3], [4,5,6]], [-1,])
```




    array([1, 2, 3, 4, 5, 6])




```python
# define optimizer operation to minimize the loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(decoder.loss)

# will be used to save/load network weights.
# you need to reset your default graph and define it in the same way to be able to load the saved weights!
saver = tf.train.Saver()

# intialize all variables
s.run(tf.global_variables_initializer())
```

    /usr/local/lib/python3.5/dist-packages/tensorflow/python/ops/gradients_impl.py:108: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
      "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "


## Training loop
Evaluate train and validation metrics through training and log them. Ensure that loss decreases.


```python
train_captions_indexed = np.array(train_captions_indexed)
val_captions_indexed = np.array(val_captions_indexed)
```


```python
train_captions_indexed[0]
```




    [[2, 54, 4462, 2305, 6328, 3354, 7848, 54, 3107, 0],
     [2, 54, 6540, 5127, 8486, 249, 54, 5437, 8507, 0],
     [2, 54, 6502, 5437, 7581, 1124, 8052, 4287, 7905, 54, 639, 0],
     [2, 5120, 8367, 7848, 54, 2305, 6328, 7581, 1124, 54, 3108, 331, 0],
     [2, 2305, 5437, 4286, 710, 587, 8052, 7905, 5174, 5684, 0]]




```python
# generate batch via random sampling of images and captions for them,
# we use `max_len` parameter to control the length of the captions (truncating long captions)
def generate_batch(images_embeddings, indexed_captions, batch_size, max_len=None):
    """
    `images_embeddings` is a np.array of shape [number of images, IMG_EMBED_SIZE].
    `indexed_captions` holds 5 vocabulary indexed captions for each image:
    [
        [
            [vocab[START], vocab["image1"], vocab["caption1"], vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"], vocab[END]],
            ...
        ],
        ...
    ]
    Generate a random batch of size `batch_size`.
    Take random images and choose one random caption for each image.
    Remember to use `batch_captions_to_matrix` for padding and respect `max_len` parameter.
    Return feed dict {decoder.img_embeds: ..., decoder.sentences: ...}.
    """
    # here images_embeddings and indexed_captions corresponds row by row
    image_idx = np.random.choice(range(len(images_embeddings)), batch_size, replace=False)
    batch_image_embeddings = images_embeddings[image_idx]
    batch_captions = [cap[np.random.randint(len(cap))] for cap in indexed_captions[image_idx]]
    # need padding to ensure the length of the captioning sentences are the same
    batch_captions_matrix = batch_captions_to_matrix(batch_captions, pad_idx, max_len=max_len)
    return {decoder.img_embeds: batch_image_embeddings, 
            decoder.sentences: batch_captions_matrix}
```


```python
batch_size = 64
n_epochs = 12
n_batches_per_epoch = 1000
n_validation_batches = 100  # how many batches are used for validation after each epoch
```


```python
# you can load trained weights here
# you can load "weights_{epoch}" and continue training
# uncomment the next line if you need to load weights
saver.restore(s, os.path.abspath("weights"))
```

    INFO:tensorflow:Restoring parameters from /home/karen/workspace/Advanced_ML_HSE/Introduction-To-Deep-Learning/notebooks/week6/weights



```python
MAX_LEN = 20  # truncate long captions to speed up training
np.random.seed(42)
random.seed(42)

for epoch in range(n_epochs):
    train_loss = 0
    pbar = tqdm.tqdm_notebook(range(n_batches_per_epoch))
    counter = 0
    for _ in pbar:
        train_loss += s.run([decoder.loss, train_step], 
                            generate_batch(train_img_embeds, 
                                           train_captions_indexed, 
                                           batch_size, 
                                           MAX_LEN))[0]
        counter += 1
        pbar.set_description("Training loss: %f" % (train_loss / counter))
        
    train_loss /= n_batches_per_epoch
    
    val_loss = 0
    for _ in range(n_validation_batches):
        val_loss += s.run(decoder.loss, generate_batch(val_img_embeds,
                                                       val_captions_indexed, 
                                                       batch_size, 
                                                       MAX_LEN))
    val_loss /= n_validation_batches
    
    print('Epoch: {}, train loss: {}, val loss: {}'.format(epoch, train_loss, val_loss))

    # save weights after finishing epoch
    saver.save(s, os.path.abspath("weights_{}".format(epoch)))
    
print("Finished!")
```

    
    
    Epoch: 0, train loss: 4.310093719482422, val loss: 3.653768141269684
    
    Epoch: 1, train loss: 3.3454977943897246, val loss: 3.1484389114379883
    
    Epoch: 2, train loss: 3.045226105928421, val loss: 2.992725837230682



```python
# check that it's learnt something, outputs accuracy of next word prediction (should be around 0.5)
from sklearn.metrics import accuracy_score, log_loss

def decode_sentence(sentence_indices):
    # vocab_inverse => idx:word
    return " ".join(list(map(vocab_inverse.get, sentence_indices)))

def check_after_training(n_examples):
    fd = generate_batch(train_img_embeds, train_captions_indexed, batch_size)
    logits = decoder.flat_token_logits.eval(fd)
    print(logits)
    truth = decoder.flat_ground_truth.eval(fd)
    mask = decoder.flat_loss_mask.eval(fd).astype(bool)
    print("Loss:", decoder.loss.eval(fd))
    print("Accuracy:", accuracy_score(logits.argmax(axis=1)[mask], truth[mask]))
    for example_idx in range(n_examples):
        print("Example", example_idx)
        print("Predicted:", decode_sentence(logits.argmax(axis=1).reshape((batch_size, -1))[example_idx]))
        print("Truth:", decode_sentence(truth.reshape((batch_size, -1))[example_idx]))
        print("")

check_after_training(3)
```

    [[ -1.0632595 -10.616826  -10.42775   ...  -5.4132915  -4.970221
       -3.9142747]
     [ -3.4125242 -10.622873  -10.226078  ...  -6.121539   -4.0196548
       -4.258054 ]
     [  1.9395864  -9.076095   -8.886731  ...  -5.375271    0.4434785
       -5.1517363]
     ...
     [  8.847406  -10.174876   -9.661328  ...  -4.5233927  -4.2720203
       -5.4303093]
     [  8.821748  -10.143098   -9.6292715 ...  -4.512379   -4.2465806
       -5.4106746]
     [  8.803145  -10.118174   -9.603903  ...  -4.5036845  -4.2257204
       -5.3926725]]
    Loss: 2.4603822
    Accuracy: 0.4736129905277402
    Example 0
    Predicted: a traffic is driving down on a traffic light #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END#
    Truth: a car is turning left at a street intersection #END# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD#
    
    Example 1
    Predicted: a cat sitting sitting on a desk with with a #END# #END# a a laptop bear #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END#
    Truth: a man is sitting at his desk playing on #UNK# #UNK# and holding a teddy bear #END# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD#
    
    Example 2
    Predicted: a living room with a couch and and table and a chairs #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END# #END#
    Truth: a living room with a couch pillows a table and some books #END# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD# #PAD#
    



```python
# save graph weights to file!
saver.save(s, os.path.abspath("weights"))
```




    '/home/karen/workspace/Advanced_ML_HSE/Introduction-To-Deep-Learning/notebooks/week6/weights'



# Applying model

Here we construct a graph for our final model.

It will work as follows:
- take an image as an input and embed it
- condition lstm on that embedding
- predict the next token given a START input token
- use predicted token as an input at next time step
- iterate until you predict an END token


```python
class final_model:
    # CNN encoder
    encoder, preprocess_for_model = get_cnn_encoder()
    saver.restore(s, os.path.abspath("weights"))  # keras applications corrupt our graph, so we restore trained weights
    
    # containers for current lstm state
    lstm_c = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="cell")
    lstm_h = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="hidden")

    # input images
    input_images = tf.placeholder('float32', [1, IMG_SIZE, IMG_SIZE, 3], name='images')

    # get image embeddings
    img_embeds = encoder(input_images)

    # initialize lstm state conditioned on image
    init_c = init_h = decoder.img_embed_bottleneck_to_h0(decoder.img_embed_to_bottleneck(img_embeds))
    init_lstm = tf.assign(lstm_c, init_c), tf.assign(lstm_h, init_h)
    
    # current word index
    current_word = tf.placeholder('int32', [1], name='current_input')

    # embedding for current word
    word_embed = decoder.word_embed(current_word)

    # apply lstm cell, get new lstm states
    new_c, new_h = decoder.lstm(word_embed, tf.nn.rnn_cell.LSTMStateTuple(lstm_c, lstm_h))[1]

    # compute logits for next token
    new_logits = decoder.token_logits(decoder.token_logits_bottleneck(new_h))
    # compute probabilities for next token
    new_probs = tf.nn.softmax(new_logits)

    # `one_step` outputs probabilities of next token and updates lstm hidden state
    one_step = new_probs, tf.assign(lstm_c, new_c), tf.assign(lstm_h, new_h)
```

    INFO:tensorflow:Restoring parameters from /home/karen/workspace/Advanced_ML_HSE/Introduction-To-Deep-Learning/notebooks/week6/weights



```python
# look at how temperature works for probability distributions
# for high temperature we have more uniform distribution
_ = np.array([0.5, 0.4, 0.1])
for t in [0.01, 0.1, 1, 10, 100]:
    print(" ".join(map(str, _**(1/t) / np.sum(_**(1/t)))), "with temperature", t)
```

    0.9999999997962965 2.0370359759195462e-10 1.2676505999700117e-70 with temperature 0.01
    0.9030370433250645 0.09696286420394223 9.247099323648666e-08 with temperature 0.1
    0.5 0.4 0.1 with temperature 1
    0.35344772639219624 0.34564811360592396 0.3009041600018798 with temperature 10
    0.33536728048099185 0.33461976434857876 0.3300129551704294 with temperature 100



```python
# this is an actual prediction loop
def generate_caption(image, t=1, sample=False, max_len=20):
    """
    Generate caption for given image.
    if `sample` is True, we will sample next token from predicted probability distribution.
    `t` is a temperature during that sampling,
        higher `t` causes more uniform-like distribution = more chaos.
    """
    # condition lstm on the image
    s.run(final_model.init_lstm, 
          {final_model.input_images: [image]})
    
    # current caption
    # start with only START token
    caption = [vocab[START]]
    
    for _ in range(max_len):
        next_word_probs = s.run(final_model.one_step, 
                                {final_model.current_word: [caption[-1]]})[0]
        next_word_probs = next_word_probs.ravel()
        
        # apply temperature
        next_word_probs = next_word_probs**(1/t) / np.sum(next_word_probs**(1/t))

        if sample:
            next_word = np.random.choice(range(len(vocab)), p=next_word_probs)
        else:
            next_word = np.argmax(next_word_probs)

        caption.append(next_word)
        if next_word == vocab[END]:
            break
       return list(map(vocab_inverse.get, caption))
```


```python
# look at validation prediction example
def apply_model_to_image_raw_bytes(raw):
    img = utils.decode_image_from_buf(raw)
    fig = plt.figure(figsize=(7, 7))
    plt.grid('off')
    plt.axis('off')
    plt.imshow(img)
    img = utils.crop_and_preprocess(img, (IMG_SIZE, IMG_SIZE), final_model.preprocess_for_model)
    print(' '.join(generate_caption(img)[1:-1]))
    plt.show()

def show_valid_example(val_img_fns, example_idx=0):
    zf = zipfile.ZipFile("/home/karen/Downloads/data/val2014_sample.zip")
    all_files = set(val_img_fns)
    found_files = list(filter(lambda x: x.filename.rsplit("/")[-1] in all_files, zf.filelist))
    example = found_files[example_idx]
    apply_model_to_image_raw_bytes(zf.read(example))
    
show_valid_example(val_img_fns, example_idx=100)
```

    /usr/local/lib/python3.5/dist-packages/matplotlib/cbook/deprecation.py:107: MatplotlibDeprecationWarning: Passing one of 'on', 'true', 'off', 'false' as a boolean is deprecated; use an actual boolean (True/False) instead.
      warnings.warn(message, mplDeprecation, stacklevel=1)


    a baseball player swinging a bat at a baseball game



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week6/output_67_2.png)



```python
# sample more images from validation
for idx in np.random.choice(range(len(zipfile.ZipFile("/home/karen/Downloads/data/val2014_sample.zip").filelist) - 1), 10):
    show_valid_example(val_img_fns, example_idx=idx)
    time.sleep(1)
```

    /usr/local/lib/python3.5/dist-packages/matplotlib/cbook/deprecation.py:107: MatplotlibDeprecationWarning: Passing one of 'on', 'true', 'off', 'false' as a boolean is deprecated; use an actual boolean (True/False) instead.
      warnings.warn(message, mplDeprecation, stacklevel=1)


    a white plate with a piece of cake and a fork



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week6/output_68_2.png)


    a group of people standing on top of a boat



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week6/output_68_4.png)


    a woman is walking down a street holding a bag of luggage



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week6/output_68_6.png)


    a street sign with a street sign and a street sign



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week6/output_68_8.png)


    a pair of scissors are on a wall



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week6/output_68_10.png)


    a large jetliner sitting on top of a tarmac



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week6/output_68_12.png)


    a young boy holding a baby in a bowl



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week6/output_68_14.png)


    a cat sitting on a chair next to a cat



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week6/output_68_16.png)


    a bathroom with a toilet and a sink



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week6/output_68_18.png)


    a man on a beach with a surfboard in the water



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week6/output_68_20.png)


You can download any image from the Internet and appply your model to it!


```python
import download_utils
```


```python
download_utils.download_file(
    "https://img.freepik.com/free-photo/positive-businessman-using-laptop_23-2147800028.jpg",
    "man_and_laptop.jpg"
)
```


    HBox(children=(IntProgress(value=0, max=68504), HTML(value='')))



```python
download_utils.download_file(url="https://thumbs.dreamstime.com/z/young-man-feeding-dog-sitting-floor-room-est-home-relaxation-concepts-people-pets-relaxed-man-white-dog-puppy-room-123726557.jpg",
                            file_path="man_feeding_dog.jpg")
```


    HBox(children=(IntProgress(value=0, max=103857), HTML(value='')))



```python
apply_model_to_image_raw_bytes(open("man_feeding_dog.jpg", "rb").read())
```

    /usr/local/lib/python3.5/dist-packages/matplotlib/cbook/deprecation.py:107: MatplotlibDeprecationWarning: Passing one of 'on', 'true', 'off', 'false' as a boolean is deprecated; use an actual boolean (True/False) instead.
      warnings.warn(message, mplDeprecation, stacklevel=1)


    a man sitting on a couch with a dog



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week6/output_73_2.png)



```python
apply_model_to_image_raw_bytes(open("man_and_laptop.jpg", "rb").read())
```

    /usr/local/lib/python3.5/dist-packages/matplotlib/cbook/deprecation.py:107: MatplotlibDeprecationWarning: Passing one of 'on', 'true', 'off', 'false' as a boolean is deprecated; use an actual boolean (True/False) instead.
      warnings.warn(message, mplDeprecation, stacklevel=1)


    a man sitting at a table with a laptop computer



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week6/output_74_2.png)


Now it's time to find 10 examples where your model works good and 10 examples where it fails! 

You can use images from validation set as follows:
```python
show_valid_example(val_img_fns, example_idx=...)
```

You can use images from the Internet as follows:
```python
! wget ...
apply_model_to_image_raw_bytes(open("...", "rb").read())
```

If you use these functions, the output will be embedded into your notebook and will be visible during peer review!
