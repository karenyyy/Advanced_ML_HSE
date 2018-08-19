
# MNIST digits classification with TensorFlow


```python
from matplotlib import pyplot as plt
from IPython.display import clear_output, display_html, HTML
import contextlib
import time
import io
import urllib
import base64


def clear_and_display_figure(fig, sleep=0.01):
    img_data = io.BytesIO()
    fig.savefig(img_data, format='jpeg')
    img_data.seek(0)
    uri = 'data:image/jpeg;base64,' + urllib.request.quote(base64.b64encode(img_data.getbuffer()))
    img_data.close()
    clear_output(wait=True)
    display_html(HTML(''))
    time.sleep(sleep)


class SimpleMovieWriter(object):
    def __init__(self, sleep=0.1):
        self.sleep = sleep

    def setup(self, fig):
        self.fig = fig

    def grab_frame(self, **kwargs):
        clear_and_display_figure(self.fig, self.sleep)

    @contextlib.contextmanager
    def saving(self, fig, *args, **kwargs):
        self.setup(fig)
        try:
            yield self
        finally:
            pass


class SimpleTrainingCurves(object):
    def __init__(self, loss_name, metric_name):
        self.fig, (self.ax1, self.ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        self.ax1.set_title(loss_name)
        self.ax2.set_title(metric_name)
        self.train_loss_curve, = self.ax1.plot([], [], 'r', label='train', lw=2)
        self.valid_loss_curve, = self.ax1.plot([], [], 'g', label='valid', lw=2)
        self.train_metric_curve, = self.ax2.plot([], [], 'r', label='train', lw=2)
        self.valid_metric_curve, = self.ax2.plot([], [], 'g', label='valid', lw=2)
        self.iter = 0
        self.y_limits_1 = [None, None]
        self.y_limits_2 = [None, None]
        plt.close(self.fig)

    def _update_y_limits(self, limits, *values):
        limits[0] = min(list(values) + ([limits[0]] if limits[0] else []))
        limits[1] = max(list(values) + ([limits[1]] if limits[1] else []))

    def _update_curve(self, curve, value, label):
        x, y = curve.get_data()
        curve.set_data(list(x) + [self.iter], list(y) + [value])
        curve.set_label("{}: {}".format(label, value))

    def _set_y_limits(self, ax, limits):
        spread = limits[1] - limits[0]
        ax.set_ylim(limits[0] - 0.05*spread, limits[1] + 0.05*spread)

    def add(self, train_loss, valid_loss, train_metric, valid_metric):
        self._update_curve(self.train_loss_curve, train_loss, "train")
        self._update_curve(self.valid_loss_curve, valid_loss, "valid")
        self._update_curve(self.train_metric_curve, train_metric, "train")
        self._update_curve(self.valid_metric_curve, valid_metric, "valid")
        self.ax1.set_xlim(0, self.iter)
        self.ax2.set_xlim(0, self.iter)
        self._update_y_limits(self.y_limits_1, train_loss, valid_loss)
        self._update_y_limits(self.y_limits_2, train_metric, valid_metric)
        self._set_y_limits(self.ax1, self.y_limits_1)
        self._set_y_limits(self.ax2, self.y_limits_2)
        clear_and_display_figure(self.fig)
        self.ax1.legend()
        self.ax2.legend()
        self.iter += 1
```


```python
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

import tensorflow as tf

import sys
sys.path.append("../..")

from importlib import reload

```

# Fill in your Coursera token and email
To successfully submit your answers to our grader, please fill in your Coursera submission token and email


```python
grader = grading.Grader(assignment_key="XtD7ho3TEeiHQBLWejjYAA", 
                        all_parts=["9XaAS", "vmogZ", "RMv95", "i8bgs", "rE763"])
```


```python
# token expires every 30 min
COURSERA_TOKEN = "U7Fcor4WWrWBXxDJ"
COURSERA_EMAIL = "karenye.psu@gmail.com"
```

# Look at the data

In this task we have 50000 28x28 images of digits from 0 to 9.
We will train a classifier on this data.


```python
import keras

def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test

```


```python
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
```


```python
# X contains rgb values divided by 255
print("X_train [shape %s] sample patch:\n" % (str(X_train.shape)), X_train[1, 15:20, 5:10])
print("A closeup of a sample patch:")
plt.imshow(X_train[1, 15:20, 5:10], cmap="Greys")
plt.show()
print("And the whole sample:")
plt.imshow(X_train[1], cmap="Greys")
plt.show()
print("y_train [shape %s] 10 samples:\n" % (str(y_train.shape)), y_train[:10])
```

    X_train [shape (50000, 28, 28)] sample patch:
     [[ 0.          0.29803922  0.96470588  0.98823529  0.43921569]
     [ 0.          0.33333333  0.98823529  0.90196078  0.09803922]
     [ 0.          0.33333333  0.98823529  0.8745098   0.        ]
     [ 0.          0.33333333  0.98823529  0.56862745  0.        ]
     [ 0.          0.3372549   0.99215686  0.88235294  0.        ]]
    A closeup of a sample patch:



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week2/output_9_1.png)


    And the whole sample:



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week2/output_9_3.png)


    y_train [shape (50000,)] 10 samples:
     [5 0 4 1 9 2 1 3 1 4]


# Linear model

Your task is to train a linear classifier $\vec{x} \rightarrow y$ with SGD using TensorFlow.

You will need to calculate a logit (a linear transformation) $z_k$ for each class: 
$$z_k = \vec{x} \cdot \vec{w_k} + b_k \quad k = 0..9$$

And transform logits $z_k$ to valid probabilities $p_k$ with softmax: 
$$p_k = \frac{e^{z_k}}{\sum_{i=0}^{9}{e^{z_i}}} \quad k = 0..9$$

We will use a cross-entropy loss to train our multi-class classifier:
$$\text{cross-entropy}(y, p) = -\sum_{k=0}^{9}{\log(p_k)[y = k]}$$ 

where 
$$
[x]=\begin{cases}
       1, \quad \text{if $x$ is true} \\
       0, \quad \text{otherwise}
    \end{cases}
$$

Cross-entropy minimization pushes $p_k$ close to 1 when $y = k$, which is what we want.

Here's the plan:
* Flatten the images (28x28 -> 784) with `X_train.reshape((X_train.shape[0], -1))` to simplify our linear model implementation
* Use a matrix placeholder for flattened `X_train`
* Convert `y_train` to one-hot encoded vectors that are needed for cross-entropy
* Use a shared variable `W` for all weights (a column $\vec{w_k}$ per class) and `b` for all biases.
* Aim for ~0.93 validation accuracy


```python
X_train_flat = X_train.reshape((X_train.shape[0], -1))
print(X_train_flat.shape)

X_val_flat = X_val.reshape((X_val.shape[0], -1))
print(X_val_flat.shape)
```

    (50000, 784)
    (10000, 784)



```python
import keras

y_train_oh = keras.utils.to_categorical(y_train, 10)
y_val_oh = keras.utils.to_categorical(y_val, 10)

print(y_train_oh.shape)
print(y_train_oh[:3], y_train[:3])
```

    (50000, 10)
    [[ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
     [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]] [5 0 4]



```python
# run this if you remake your graph
tf.reset_default_graph()
```


```python
# Model parameters: W and b
W = tf.get_variable('weight', [X_train_flat.shape[1], 10], dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b = tf.get_variable('biase', [10], dtype=tf.float32, initializer=tf.zeros_initializer())
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    RuntimeError: module compiled against API version 0xb but this version of numpy is 0xa



```python
# Placeholders for the input data
input_X = tf.placeholder(dtype=tf.float32, shape=(None, X_train_flat.shape[1]), name='X')
input_y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='y')
```


```python
# Compute predictions
logits = tf.add(tf.matmul(input_X, W), b)
probas = tf.nn.softmax(logits)
classes = tf.argmax(probas, 1)

# Loss should be a scalar number: average loss over all the objects with tf.reduce_mean().
# Use tf.nn.softmax_cross_entropy_with_logits on top of one-hot encoded input_y and logits.
# It is identical to calculating cross-entropy on top of probas, but is more numerically friendly (read the docs).
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y))

# Use a default tf.train.AdamOptimizer to get an SGD step
step = tf.train.AdamOptimizer(learning_rate=0.004).minimize(loss)
```


```python
s = tf.InteractiveSession()
s.run(tf.global_variables_initializer())

BATCH_SIZE = 512
EPOCHS = 40

# for logging the progress right here in Jupyter (for those who don't have TensorBoard)
simpleTrainingCurves = matplotlib_utils.SimpleTrainingCurves("cross-entropy", "accuracy")

for epoch in range(EPOCHS):  # we finish an epoch when we've looked at all training samples
    
    batch_losses = []
    for batch_start in range(0, X_train_flat.shape[0], BATCH_SIZE):  # data is already shuffled
        _, batch_loss = s.run([step, loss], {input_X: X_train_flat[batch_start:batch_start+BATCH_SIZE], 
                                             input_y: y_train_oh[batch_start:batch_start+BATCH_SIZE]})
        # collect batch losses, this is almost free as we need a forward pass for backprop anyway
        batch_losses.append(batch_loss)
    train_loss = np.mean(batch_losses)
    val_loss = s.run(loss, {input_X: X_val_flat, input_y: y_val_oh})  # this part is usually small
    train_accuracy = accuracy_score(y_train, s.run(classes, {input_X: X_train_flat}))  # this is slow and usually skipped
    valid_accuracy = accuracy_score(y_val, s.run(classes, {input_X: X_val_flat}))  
    simpleTrainingCurves.add(train_loss, val_loss, train_accuracy, valid_accuracy)
```



![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week2/1.png)


# Submit a linear model


```python
## GRADED PART, DO NOT CHANGE!
# Testing shapes 
grader.set_answer("9XaAS", grading_utils.get_tensors_shapes_string([W, b, input_X, input_y, logits, probas, classes]))
# Validation loss
grader.set_answer("vmogZ", s.run(loss, {input_X: X_val_flat, input_y: y_val_oh}))
# Validation accuracy
grader.set_answer("RMv95", accuracy_score(y_val, s.run(classes, {input_X: X_val_flat})))
```


```python
# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)
```

    Submitted to Coursera platform. See results on assignment page!


# Adding more layers

Let's add a couple of hidden layers and see how that improves our validation accuracy.

Previously we've coded a dense layer with matrix multiplication by hand. 
But this is not convinient, you have to create a lot of variables and your code becomes a mess. 
In TensorFlow there's an easier way to make a dense layer:
```python
hidden1 = tf.layers.dense(inputs, 256, activation=tf.nn.sigmoid)
```

That will create all the necessary variables automatically.
Here you can also choose an activation function (rememeber that we need it for a hidden layer!).

Now add 2 hidden layers to the code above and restart training.
You're aiming for ~0.97 validation accuracy here.


```python
# you can write the code here to get a new `step` operation and then run cells with training loop above
# name your variables in the same way (e.g. logits, probas, classes, etc) for safety
def init_placeholder():
    input_X = tf.placeholder(dtype=tf.float32, shape=(None, X_train_flat.shape[1]), name='X')
    input_y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='y')
    return input_X, input_y

def init_parameters():
    W1 = tf.get_variable('w1', [X_train_flat.shape[1], 256], dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable('b1', [256], dtype=tf.float32, initializer=tf.zeros_initializer())
    W2 = tf.get_variable('w2', [256, 256], dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable('b2', [256], dtype=tf.float32, initializer=tf.zeros_initializer())
    W3 = tf.get_variable('w3', [256, 10], dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable('b3', [10], dtype=tf.float32, initializer=tf.zeros_initializer())
    parameters = {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2, 'W3':W3, 'b3':b3}
    return parameters

def model(parameters, X):
    layer1 = tf.add(tf.matmul(X, parameters['W1']), parameters['b1'])
    act1 = tf.nn.relu(layer1)
    # act1 = tf.nn.dropout(act1, 0.5)
    layer2 = tf.add(tf.matmul(act1, parameters['W2']), parameters['b2'])
    act2 = tf.nn.relu(layer2)
    # act2 = tf.nn.dropout(act2, 0.5)
    layer3 = tf.add(tf.matmul(act2, parameters['W3']), parameters['b3'])
    return layer3

def compute_loss(logits, y, W1, W2, beta=0.0001): 
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
    loss = tf.reduce_mean(loss + beta * regularizer)
    return loss

def optimizer(loss):
    step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    return step


   
tf.reset_default_graph()
input_X, input_y = init_placeholder()
parameters = init_parameters()

BATCH_SIZE = 512
EPOCHS = 40

logits = model(parameters=parameters, X = input_X)
probas = tf.nn.softmax(logits)
classes = tf.argmax(probas, 1)
loss = compute_loss(logits=logits, y = input_y, W1 = parameters['W1'], W2 = parameters['W2'])
step = optimizer(loss)
simpleTrainingCurves = matplotlib_utils.SimpleTrainingCurves("cross-entropy", "accuracy")
    
s = tf.InteractiveSession()
s.run(tf.global_variables_initializer())

for epoch in range(EPOCHS):
    batch_losses = []
    for batch_start in range(0, X_train_flat.shape[0], BATCH_SIZE): 
        _, batch_loss = s.run([step, loss], {input_X: X_train_flat[batch_start:batch_start+BATCH_SIZE], 
                                                  input_y: y_train_oh[batch_start:batch_start+BATCH_SIZE]})
        
        batch_losses.append(batch_loss)
    train_loss = np.mean(batch_losses)
    val_loss = s.run(loss, {input_X: X_val_flat, input_y: y_val_oh})  
    train_accuracy = accuracy_score(y_train, s.run(classes, {input_X: X_train_flat}))  
    valid_accuracy = accuracy_score(y_val, s.run(classes, {input_X: X_val_flat}))  
    simpleTrainingCurves.add(train_loss, val_loss, train_accuracy, valid_accuracy)
```


![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week2/2.jpeg)


# Submit a 2-layer MLP
Run these cells after training a 2-layer MLP


```python
## GRADED PART, DO NOT CHANGE!
# Validation loss for MLP
grader.set_answer("i8bgs", s.run(loss, {input_X: X_val_flat, input_y: y_val_oh}))
# Validation accuracy for MLP
grader.set_answer("rE763", accuracy_score(y_val, s.run(classes, {input_X: X_val_flat})))
```


```python
# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)
```

    Submitted to Coursera platform. See results on assignment page!
 

## MNIST with Keras


```python
def reset_tf_session():
    keras.backend.clear_session()
    tf.reset_default_graph()
    s = keras.backend.get_session()
    return s
```


```python
# building a model with keras
from keras.layers import Dense, Activation
from keras.models import Sequential

s = reset_tf_session()

model = Sequential()  # it is a feed-forward network without loops like in RNN
model.add(Dense(256, input_shape=(784,)))  # the first layer must specify the input shape (replacing placeholders)
model.add(Activation('sigmoid'))
model.add(Dense(256))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 256)               200960    
    _________________________________________________________________
    activation_1 (Activation)    (None, 256)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 256)               65792     
    _________________________________________________________________
    activation_2 (Activation)    (None, 256)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                2570      
    _________________________________________________________________
    activation_3 (Activation)    (None, 10)                0         
    =================================================================
    Total params: 269,322
    Trainable params: 269,322
    Non-trainable params: 0
    _________________________________________________________________



```python
model.compile(
    loss='categorical_crossentropy', # this is our cross-entropy
    optimizer='adam',
    metrics=['accuracy']  # report accuracy during training
)
history = model.fit(
    X_train_flat, 
    y_train_oh,
    batch_size=512, 
    epochs=40,
    validation_data=(X_val_flat, y_val_oh),
    verbose=0
)
```


```python
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

    dict_keys(['acc', 'loss', 'val_loss', 'val_acc'])



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week2/output_31_1.png)



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week2/output_31_2.png)

