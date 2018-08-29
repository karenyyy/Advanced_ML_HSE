
### Generating human faces with Adversarial Networks

This time we'll train a neural net to generate plausible human faces in all their subtlty: appearance, expression, accessories, etc. 'Cuz when us machines gonna take over Earth, there won't be any more faces left. We want to preserve this data for future iterations. Yikes...

Based on Based on https://github.com/Lasagne/Recipes/pull/94 .



```python
import sys
sys.path.append("..")
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
plt.rcParams.update({'axes.titlesize': 'small'})

from sklearn.datasets import load_digits
#The following line fetches you two datasets: images, usable for autoencoder training and attributes.
#Those attributes will be required for the final part of the assignment (applying smiles), so please keep them in mind
from lfw_dataset import load_lfw_dataset 
data,attrs = load_lfw_dataset(dimx=36,dimy=36)

#preprocess faces
data = np.float32(data)/255.

IMG_SHAPE = data.shape[1:]
```


<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    



```python
#print random image
plt.imshow(data[np.random.randint(data.shape[0])], cmap="gray", interpolation="none")
```




    <matplotlib.image.AxesImage at 0x7f012cd019b0>




![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_3_1.png)


# Generative adversarial nets 101

<img src="https://raw.githubusercontent.com/torch/torch.github.io/master/blog/_posts/images/model.png" width=320px height=240px>

Deep learning is simple, isn't it? 
* build some network that generates the face (small image)
* make up a __measure__ of __how good that face is__
* optimize with gradient descent :)


The only problem is: how can we engineers tell well-generated faces from bad? And i bet you we won't ask a designer for help. 

__If we can't tell good faces from bad, we delegate it to yet another neural network!__

That makes the two of them:
* __G__enerator - takes random noize for inspiration and tries to generate a face sample. 
  * Let's call him __G__(z), where z is a gaussian noize.
* __D__iscriminator - takes a face sample and tries to tell if it's great or fake. 
  * Predicts the probability of input image being a __real face__
  * Let's call him __D__(x), x being an image.
  * __D(x)__ is a predition for real image and __D(G(z))__ is prediction for the face made by generator.

Before we dive into training them, let's construct the two networks.


```python
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True,per_process_gpu_memory_fraction=0.333)
s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

import keras
from keras.models import Sequential
from keras import layers as L
```

    /home/karen/.local/lib/python3.5/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.



```python
CODE_SIZE = 256

generator = Sequential()
generator.add(L.InputLayer([CODE_SIZE],name='noise'))
generator.add(L.Dense(10*8*8, activation='elu'))

generator.add(L.Reshape((8,8,10)))
generator.add(L.Deconv2D(64,kernel_size=(5,5),activation='elu'))
generator.add(L.Deconv2D(64,kernel_size=(5,5),activation='elu'))
generator.add(L.UpSampling2D(size=(2,2)))
generator.add(L.Deconv2D(32,kernel_size=3,activation='elu'))
generator.add(L.Deconv2D(32,kernel_size=3,activation='elu'))
generator.add(L.Deconv2D(32,kernel_size=3,activation='elu'))

generator.add(L.Conv2D(3,kernel_size=3,activation=None))

```


```python
assert generator.output_shape[1:] == IMG_SHAPE, "generator must output an image of shape %s, but instead it produces %s"%(IMG_SHAPE,generator.output_shape[1:])
```


```python
IMG_SHAPE
```




    (36, 36, 3)



### Discriminator
* Discriminator is your usual convolutional network with interlooping convolution and pooling layers
* The network does not include dropout/batchnorm to avoid learning complications.
* We also regularize the pre-output layer to prevent discriminator from being too certain.


```python
discriminator = Sequential()

discriminator.add(L.InputLayer(IMG_SHAPE))

discriminator.add(L.Conv2D(8, (3, 3)))
discriminator.add(L.LeakyReLU(0.1))
discriminator.add(L.Conv2D(16, (3, 3)))
discriminator.add(L.LeakyReLU(0.1))
discriminator.add(L.MaxPool2D())
discriminator.add(L.Conv2D(32, (3, 3)))
discriminator.add(L.LeakyReLU(0.1))
discriminator.add(L.Conv2D(64, (3, 3)))
discriminator.add(L.LeakyReLU(0.1))
discriminator.add(L.MaxPool2D())

discriminator.add(L.Flatten())
discriminator.add(L.Dense(256,activation='tanh'))
discriminator.add(L.Dense(2,activation=tf.nn.log_softmax))

```

    WARNING:tensorflow:From /home/karen/.local/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:1264: calling reduce_prod (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
    Instructions for updating:
    keep_dims is deprecated, use keepdims instead



```python
discriminator.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 36, 36, 3)         0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 34, 34, 8)         224       
    _________________________________________________________________
    leaky_re_lu_1 (LeakyReLU)    (None, 34, 34, 8)         0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 32, 32, 16)        1168      
    _________________________________________________________________
    leaky_re_lu_2 (LeakyReLU)    (None, 32, 32, 16)        0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 16, 16, 16)        0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 14, 14, 32)        4640      
    _________________________________________________________________
    leaky_re_lu_3 (LeakyReLU)    (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 12, 12, 64)        18496     
    _________________________________________________________________
    leaky_re_lu_4 (LeakyReLU)    (None, 12, 12, 64)        0         
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 6, 6, 64)          0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 2304)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 256)               590080    
    _________________________________________________________________
    dense_3 (Dense)              (None, 2)                 514       
    =================================================================
    Total params: 615,122
    Trainable params: 615,122
    Non-trainable params: 0
    _________________________________________________________________


# Training

We train the two networks concurrently:
* Train __discriminator__ to better distinguish real data from __current__ generator
* Train __generator__ to make discriminator think generator is real
* Since discriminator is a differentiable neural network, we train both with gradient descent.

![img](https://s24.postimg.org/cw4nognxx/gan.png)

Training is done iteratively until discriminator is no longer able to find the difference (or until you run out of patience).


### Tricks:
* Regularize discriminator output weights to prevent explosion
* Train generator with __adam__ to speed up training. Discriminator trains with SGD to avoid problems with momentum.
* More: https://github.com/soumith/ganhacks



```python
noise = tf.placeholder('float32',[None,CODE_SIZE])
real_data = tf.placeholder('float32',[None,]+list(IMG_SHAPE))

logp_real = discriminator(real_data)

generated_data = generator(noise)

logp_gen = discriminator(generated_data)

```


```python
########################
#discriminator training#
########################

d_loss = -tf.reduce_mean(logp_real[:,1] + logp_gen[:,0])

#regularize
d_loss += tf.reduce_mean(discriminator.layers[-1].kernel**2)

#optimize
disc_optimizer =  tf.train.GradientDescentOptimizer(1e-3).minimize(d_loss,var_list=discriminator.trainable_weights)
```


```python
########################
###generator training###
########################

g_loss = -tf.reduce_mean(logp_gen[:,1])

gen_optimizer = tf.train.AdamOptimizer(1e-4).minimize(g_loss,var_list=generator.trainable_weights)

    
```


```python
s.run(tf.global_variables_initializer())
```

### Auxilary functions
Here we define a few helper functions that draw current data distributions and sample training batches.


```python
def sample_noise_batch(bsize):
    return np.random.normal(size=(bsize, CODE_SIZE)).astype('float32')

def sample_data_batch(bsize):
    idxs = np.random.choice(np.arange(data.shape[0]), size=bsize)
    return data[idxs]

def sample_images(nrow,ncol, sharp=False):
    images = generator.predict(sample_noise_batch(bsize=nrow*ncol))
    if np.var(images)!=0:
        images = images.clip(np.min(data),np.max(data))
    for i in range(nrow*ncol):
        plt.subplot(nrow,ncol,i+1)
        if sharp:
            plt.imshow(images[i].reshape(IMG_SHAPE),cmap="gray", interpolation="none")
        else:
            plt.imshow(images[i].reshape(IMG_SHAPE),cmap="gray")
    plt.show()

def sample_probas(bsize):
    plt.title('Generated vs real data')
    plt.hist(np.exp(discriminator.predict(sample_data_batch(bsize)))[:,1],
             label='D(x)', alpha=0.5,range=[0,1])
    plt.hist(np.exp(discriminator.predict(generator.predict(sample_noise_batch(bsize))))[:,1],
             label='D(G(z))',alpha=0.5,range=[0,1])
    plt.legend(loc='best')
    plt.show()
```

### Training
Main loop.
We just train generator and discriminator in a loop and draw results once every N iterations.


```python
from IPython import display
from tqdm import tnrange

for epoch in tnrange(50000):
    feed_dict = {
        real_data:sample_data_batch(100),
        noise:sample_noise_batch(100)
    }
    
    for i in range(5):
        s.run(disc_optimizer,feed_dict)
    
    s.run(gen_optimizer,feed_dict)
    
    if epoch %100==0:
        display.clear_output(wait=True)
        print("After "+ str(epoch) + " epochs")
        sample_images(2,3,True)
        sample_probas(1000)
        
```

    After 6400 epochs (6400 / 50000)
    


![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_20_1.png)



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week4/output_20_2.png)


