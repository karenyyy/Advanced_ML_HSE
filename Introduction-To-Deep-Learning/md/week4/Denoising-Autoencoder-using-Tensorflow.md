

```python
import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
```

    /usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
      return f(*args, **kwds)
    /home/karen/.local/lib/python3.5/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    /usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
      return f(*args, **kwds)
    /usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
      return f(*args, **kwds)



```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

    /usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
      return f(*args, **kwds)
    /usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
      return f(*args, **kwds)


    WARNING:tensorflow:From <ipython-input-2-8bf8ae5a5303>:2: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
    WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please write your own downloading logic.
    WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/contrib/learn/python/learn/datasets/base.py:252: _internal_retry.<locals>.wrap.<locals>.wrapped_fn (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use urllib or similar directly.
    Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
    WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
    WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.one_hot on tensors.
    Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as official/mnist/dataset.py from tensorflow/models.



```python
inputs_ = tf.placeholder(tf.float32,[None,28,28,1])
targets_ = tf.placeholder(tf.float32,[None,28,28,1])
```


```python
def lrelu(x,alpha=0.1):
    return tf.maximum(alpha*x,x)
```


```python
### Encoder
with tf.name_scope('en-convolutions'):
    conv1 = tf.layers.conv2d(inputs_,filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,activation=lrelu,name='conv1')
# Now 28x28x32
with tf.name_scope('en-pooling'):
    maxpool1 = tf.layers.max_pooling2d(conv1,pool_size=(2,2),strides=(2,2),name='pool1')
# Now 14x14x32
with tf.name_scope('en-convolutions'):
    conv2 = tf.layers.conv2d(maxpool1,filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,activation=lrelu,name='conv2')
# Now 14x14x32
with tf.name_scope('encoding'):
    encoded = tf.layers.max_pooling2d(conv2,pool_size=(2,2),strides=(2,2),name='encoding')
# Now 7x7x32.
#latent space
```


```python
### Decoder
with tf.name_scope('decoder'):
    conv3 = tf.layers.conv2d(encoded,filters=32,kernel_size=(3,3),strides=(1,1),name='conv3',padding='SAME',use_bias=True,activation=lrelu)
#Now 7x7x32        
    upsample1 = tf.layers.conv2d_transpose(conv3,filters=32,kernel_size=3,padding='same',strides=2,name='upsample1')
# Now 14x14x32
    upsample2 = tf.layers.conv2d_transpose(upsample1,filters=32,kernel_size=3,padding='same',strides=2,name='upsample2')
# Now 28x28x32
    logits = tf.layers.conv2d(upsample2,filters=1,kernel_size=(3,3),strides=(1,1),name='logits',padding='SAME',use_bias=True)
#Now 28x28x1
# Pass logits through sigmoid to get reconstructed image
    decoded = tf.sigmoid(logits,name='recon')
```


```python
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=targets_)

learning_rate=tf.placeholder(tf.float32)
cost = tf.reduce_mean(loss)  #cost
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost) #optimizer
```


```python
# Training

sess = tf.Session()
#tf.reset_default_graph()

saver = tf.train.Saver()
loss = []
valid_loss = []



display_step = 1
epochs = 25
batch_size = 64
#lr=[1e-3/(2**(i//5))for i in range(epochs)]
lr=1e-5
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./graphs', sess.graph)
for e in range(epochs):
    total_batch = int(mnist.train.num_examples/batch_size)
    for ibatch in range(total_batch):
        batch_x = mnist.train.next_batch(batch_size)
        batch_test_x= mnist.test.next_batch(batch_size)
        imgs_test = batch_x[0].reshape((-1, 28, 28, 1))
        noise_factor = 0.5
        x_test_noisy = imgs_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=imgs_test.shape) 
        x_test_noisy = np.clip(x_test_noisy, 0., 1.)
        imgs = batch_x[0].reshape((-1, 28, 28, 1))
        x_train_noisy = imgs + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=imgs.shape) 
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: x_train_noisy,
                                                         targets_: imgs,learning_rate:lr})
      
        batch_cost_test = sess.run(cost, feed_dict={inputs_: x_test_noisy,
                                                         targets_: imgs_test})
    if (e+1) % display_step == 0:
        print("Epoch: {}/{}...".format(e+1, epochs),
                  "Training loss: {:.4f}".format(batch_cost),
                 "Validation loss: {:.4f}".format(batch_cost_test))
   
    loss.append(batch_cost)
    valid_loss.append(batch_cost_test)
    plt.plot(range(e+1), loss, 'bo', label='Training loss')
    plt.plot(range(e+1), valid_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()
    saver.save(sess, 'encode_model') 

batch_x= mnist.test.next_batch(10)
imgs = batch_x[0].reshape((-1, 28, 28, 1))
noise_factor = 0.5
x_test_noisy = imgs + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=imgs.shape) 
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
recon_img = sess.run([decoded], feed_dict={inputs_: x_test_noisy})[0]
plt.figure(figsize=(20, 4))
plt.title('Reconstructed Images')
print("Original Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(imgs[i, ..., 0], cmap='gray')
plt.show()    
plt.figure(figsize=(20, 4))
print("Noisy Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test_noisy[i, ..., 0], cmap='gray')
plt.show()    
plt.figure(figsize=(20, 4))
print("Reconstruction of Noisy Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(recon_img[i, ..., 0], cmap='gray')    
plt.show()    

writer.close()

sess.close()
```

    ('Epoch: 1/25...', 'Training loss: 0.4796', 'Validation loss: 0.4800')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_1.png)



    <matplotlib.figure.Figure at 0x7f11641b6f90>


    ('Epoch: 2/25...', 'Training loss: 0.3424', 'Validation loss: 0.3400')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_4.png)



    <matplotlib.figure.Figure at 0x7f11641aa9d0>


    ('Epoch: 3/25...', 'Training loss: 0.2038', 'Validation loss: 0.2027')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_7.png)



    <matplotlib.figure.Figure at 0x7f1179254e10>


    ('Epoch: 4/25...', 'Training loss: 0.1753', 'Validation loss: 0.1770')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_10.png)



    <matplotlib.figure.Figure at 0x7f117935ea90>


    ('Epoch: 5/25...', 'Training loss: 0.1549', 'Validation loss: 0.1591')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_13.png)



    <matplotlib.figure.Figure at 0x7f11641aa8d0>


    ('Epoch: 6/25...', 'Training loss: 0.1476', 'Validation loss: 0.1477')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_16.png)



    <matplotlib.figure.Figure at 0x7f11640f2b50>


    ('Epoch: 7/25...', 'Training loss: 0.1360', 'Validation loss: 0.1382')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_19.png)



    <matplotlib.figure.Figure at 0x7f11641aa4d0>


    ('Epoch: 8/25...', 'Training loss: 0.1417', 'Validation loss: 0.1414')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_22.png)



    <matplotlib.figure.Figure at 0x7f1173912d50>


    ('Epoch: 9/25...', 'Training loss: 0.1326', 'Validation loss: 0.1333')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_25.png)



    <matplotlib.figure.Figure at 0x7f11641c0fd0>


    ('Epoch: 10/25...', 'Training loss: 0.1303', 'Validation loss: 0.1298')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_28.png)



    <matplotlib.figure.Figure at 0x7f115c5296d0>


    ('Epoch: 11/25...', 'Training loss: 0.1317', 'Validation loss: 0.1311')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_31.png)



    <matplotlib.figure.Figure at 0x7f115c5d1ed0>


    ('Epoch: 12/25...', 'Training loss: 0.1277', 'Validation loss: 0.1308')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_34.png)



    <matplotlib.figure.Figure at 0x7f1179254d90>


    ('Epoch: 13/25...', 'Training loss: 0.1249', 'Validation loss: 0.1263')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_37.png)



    <matplotlib.figure.Figure at 0x7f1177192750>


    ('Epoch: 14/25...', 'Training loss: 0.1166', 'Validation loss: 0.1171')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_40.png)



    <matplotlib.figure.Figure at 0x7f11771a1850>


    ('Epoch: 15/25...', 'Training loss: 0.1267', 'Validation loss: 0.1253')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_43.png)



    <matplotlib.figure.Figure at 0x7f1177192790>


    ('Epoch: 16/25...', 'Training loss: 0.1240', 'Validation loss: 0.1241')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_46.png)



    <matplotlib.figure.Figure at 0x7f116402b510>


    ('Epoch: 17/25...', 'Training loss: 0.1213', 'Validation loss: 0.1232')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_49.png)



    <matplotlib.figure.Figure at 0x7f115c7db3d0>


    ('Epoch: 18/25...', 'Training loss: 0.1205', 'Validation loss: 0.1200')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_52.png)



    <matplotlib.figure.Figure at 0x7f115c5d14d0>


    ('Epoch: 19/25...', 'Training loss: 0.1217', 'Validation loss: 0.1223')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_55.png)



    <matplotlib.figure.Figure at 0x7f11640f2e50>


    ('Epoch: 20/25...', 'Training loss: 0.1148', 'Validation loss: 0.1155')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_58.png)



    <matplotlib.figure.Figure at 0x7f11771a77d0>


    ('Epoch: 21/25...', 'Training loss: 0.1194', 'Validation loss: 0.1173')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_61.png)



    <matplotlib.figure.Figure at 0x7f1177192450>


    ('Epoch: 22/25...', 'Training loss: 0.1128', 'Validation loss: 0.1120')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_64.png)



    <matplotlib.figure.Figure at 0x7f117935e710>


    ('Epoch: 23/25...', 'Training loss: 0.1134', 'Validation loss: 0.1127')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_67.png)



    <matplotlib.figure.Figure at 0x7f1179368d50>


    ('Epoch: 24/25...', 'Training loss: 0.1217', 'Validation loss: 0.1214')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_70.png)



    <matplotlib.figure.Figure at 0x7f117935edd0>


    ('Epoch: 25/25...', 'Training loss: 0.1196', 'Validation loss: 0.1171')



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_73.png)



    <matplotlib.figure.Figure at 0x7f1179368350>


    Original Images



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_76.png)


    Noisy Images



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_78.png)


    Reconstruction of Noisy Images



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_7_80.png)

