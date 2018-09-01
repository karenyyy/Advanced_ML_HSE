
## Intro to Unsupervised Learning


### Autoencoder

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/9.png)

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/10.png)

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/11.png)


> Question 1:

What could possible go wrong with such autoencoder where the "code" is longer than the original data.

- __Autoencoder can learn in such a way that doesn't produce good features__ 

- Autoencoder's computation graph is invalid

- It is impossible to do backpropagation through such autoencoder

- Autoencoder can't minimize MSE as efficiently as before


![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/12.png)

#### Sparse Encoder -- use L1

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/13.png)



> Question 2:

What happens when you regularize neural network weights with L1 regularization?

L1 regularization means adding sum of absolute values of weights to the loss function.

- Some weights may end up going to infinity

- __Some weights may end up being exactly zero__

- You can no longer train your network with backprop

- Network tends to have smaller loss on training data

#### Denoizing  Autoencoder -- use Dropout

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/14.png)


#### Denoizing  Autoencoder -- use Dropout to remove input data

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/15.png)



![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/16.png)




## Natural language processing primer



![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/17.png)

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/18.png)

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/19.png)

### Embeddings

Map data into a lower dimension space while preserving structure, MDS, LLE, __TSNE__, etc.

TSNE: 

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/20.png)








### Word2Vec

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/21.png)


> Question 1:

What do you get if you compute a dot-product of a row-vector with one-hot encoding of a word[id=1337], named A, with arbitrary vector of weights named W?

- A[1337]

- A[0]

- W[0]

- __W[1337]__

- Zero

> Question 2:

What is the result of multiplying row-vector A of one-hot encoded word id1337 by a matrix W of arbitrary weights of shape [num_words, num_units]?

- __A row in W matrix, W[1337]__

- A column in W matrix, W[:,1337]

- A single number, W[1337,1337]

- Exactly the matrix W

- Exactly the vector A

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/23.png)


----
Takeaway:

- the matrix takes one-hot vector representation of one word, multiplies it by a matrix of weight
    - it is basically the idea that we have these matrix, and for each kind of mini batch, for one word, we take the corresponding row of the matrix, then send it forward along the network
 
- the second layer tries to take this representation, this word vector, to predict the neighboring words via dense layer, basically, by affine transformation

- __If two words correspond to the same context, it's beneficial for the model to assign them to similar vectors. Therefore this kind of second matrix will be able to map them into similar contexts automatically__


- We train this model by simply taking samples from the dataset, taking just sentences basically, then picking one word out of that sentence and using that word as the input, this is the middle word here. Now all other words in the sentence are considered as kind of target reference answer for our model and it tries to predict them. 

- __Once it converges, we can more or less count on this first matrix being the word representation we actually want__

----

#### This is not the only way to train word2vec. Another popular variation of this model is we could try to flip the model -- 

### Skip Gram Model -- take surrounding words and predict the middle word

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/24.png)


Which part of word2vec model would take most time to compute?

For simplicity, assume that you compute it on modern CPU as of year 2017, your vocabulary contains 100 000 words and hidden vector size is 1000

- Computing softmax given the predicted logits

- Building one-hot vector from word id

- Multiplying word vector by the right matrix

- All steps are equally computationally heavy

- __Multiplying one-hot encoded word by the left matrix__



#### More word embedding

- Faster softmax:
    - Hierarchial softmax, negative samples
- Alternative models:
    - Glove
- Sentence level:
    - Doc2vec, skip-thought(RNN)
    










## Generative Adversarial Networks

### Generative models 101

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/25.png)

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/26.png)

> Question 1

Which of those image representations is least sensitive to small shifts of objects on an image?

- Image pixels in RGB format

- Image pixels in CMYK format

- Activation of first layer of a convolutional network trained on imagenet

- __Activation of pre-final layer of a convolutional network trained on imagenet__

- Image pixels in RGB after x2 super-resolution (linear interpolation)



### Mean Squared Error

#### Pixelwise MSE

__A `cat on the left` is closer to `dog on the left` than to `cat on the right`__

we may want to avoid this effect


> Can we obtain image representation that is less sensitive to small shifts?

### Sketch: using pre-trained nets

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/27.png)


$$L = || f(img) - f(Gen(seed))||$$

----
Takeaways:

- The features in the intermediate layers usually contain elements like textures, edges, blobs, etc. When we go deep into the network, they contain high level semantic information, __EXCEPT FOR ORIENTATION AND POSITION (Introduce CapsNet in later notes)__

> The trick with position here is that it doesn't actually depend on where on your image the cat is, if we try to classify it, it's still a cat. __So it's convenient for a network to learn features that don't change 
much if the cat's position kind of changes slightly__


So we can take some intermediate layer deep enough in the network, and use the activations of the layer, apply squared error of these activations as the target metric.


__We use the previously trained classifier as another kind of specially trained metric to train a different model__






### Generative Adversarial Networks

The idea of GAN is to train a model specifically to tell us whether a generated image is good enough or not

----
#### Two Networks

- Generator
    - it inputs some kind of scene, like random noise and some parameters like object orientation and features
    - it outputs some generated images as close as possible to the groundtruth image
    - At first, the output is random noise
    - the entire generator is differentiable 
    
- Discriminator
    - discriminates between real and generated images output from the Generator
    - At first, the distinguishing is very easy because the Generator is producing rubbish (__random initialization__)


__In GAN, we train Discriminator in order to train Generator__

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/28.png)


Takeaway:
- The Discriminator is a neural network, so it's usually differentiable, so it can train 
- The goal is to tune the generator in a way that tries to fool the discriminator, and then base on the discriminator's decision, we then adjust the generative mode, so that the image becomes more real in the eye of the discriminator
- Through back propagation, we can get gradients that tell the generator how to  fool discriminator to tune the generator to produce better images to certain accuracy that the discriminator can no longer distinguish between real and generated images

> Question 2

So far we did this:

(0) initialize generator and discriminator weights at random

(1) train discriminator on to classify actual images against images generated by [untrained] generator

(2) train generator to generate images that fool discriminator into believing they're real

What's our logical next step?

- Train discriminator again on same data as before

- Train generator again to make discriminator believe it's fake again

- __Train discriminator again on images generated by updated generator__


- Train generator again on same data as before

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/29.png)

$$L_G = -\log [1-Disc(Gen(seed))] \rightarrow min$$

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/30.png)

$$L_D = -\log [1-Disc(real)] - \log Disc(Gen(seed))\rightarrow max$$

Algorithm
    - sample noise z and images x
    - for k in 1 to K
        - train `discriminator(x)`,`discriminator(generator(z))`
    - for m in 1 to M
        - train `generator(z)`


Result analysis:

- generator and discriminator compete with each other
- if one of these 2 wins, basically this means that we have to start the entire process all over again
    - if discriminator wins, then it's kind of sigmoid probability estimate of images being fake or real, it's already near 0 or near 1 which means that __the gradients vanish__ since the sigmoid has a very small variance near the activation of 1 or 0. 
    - if generator wins, i.e. if generator is constantly able to train faster than discriminator, then the situation is even worse. __Because not only does it stop learning, it starts learning all the wrong things, making it fast enough to fool discriminator, discriminator is not able to give it clear gradients of how to improve__
- Goal: must find a equilibrium!!
- Ideal scenario:
    - the whole process terminates when Generator wins after a large number of steps
- ideal generator should perfect mimic
the data distribution, be indistinguishable

    
    



### Applications of adversarial approach

Two domains
- mnist digits vs. actual digits on photos

The first domain is labeled, the second is not

Goal: to tag for the real-world dataset


![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/31.png)

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/32.png)


![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/33.png)

> Question 

If you trained a discriminator network to distinguish between features on training & test images and it easily achieved 100% accuracy: the discriminator can perfectly tell featrues(train_images) from features(test_domain_images).

Which of the following is true?

- Features can be easily discriminated, therefore they are likely very useful for the main classification problem.

- Discriminator achieves 100% accuracy, therefore the main classifier should also get good accuracy on test set.

- Classifier behaves differently on test data, therefore it overfitted to training data.


----


### Domain adaptation

__Idea: discriminator should not be able to distinguish features on two domains__



$$-\log P(real | h(x_{real})) - \log [1-P(real | h(x_{mc}))] \rightarrow \min_{discriminator}$$

$$L_{classifier} (y_{mc}, y(h(x_{mc}))) - \log P(real | h(x_{mc})) \rightarrow \min_{classifier}$$



Example: Art style transfer network structure

- Extract texture from reference image
- Extract content from original image


![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/34.png)


> Question 

How can you get features that represent an image content (i.e. what objects are there on the image) without capturing it's texture & artistic style?

- The exact same way you obtain texture features.

- RGB pixel values would do

- __Somewhere among the final layers of a CNN__








### Additional note: CapsNet (regarding the orientation and position)


```python
wdde
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-7e1710af7a56> in <module>()
    ----> 1 wdde
    

    NameError: name 'wdde' is not defined

