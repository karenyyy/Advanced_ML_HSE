
### Generating names with recurrent neural networks

This time you'll find yourself delving into the heart (and other intestines) of recurrent neural networks on a class of toy problems.

Struggle to find a name for the variable? Let's see how you'll come up with a name for your son/daughter. Surely no human has expertize over what is a good child name, so let us train RNN instead;

It's dangerous to go alone, take these:


```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

# Our data
The dataset contains ~8k earthling names from different cultures, all in latin transcript.

This notebook has been designed so as to allow you to quickly swap names for something similar: deep learning article titles, IKEA furniture, pokemon names, etc.


```python
import os
start_token = " "

with open("names") as f:
    names = f.read()[:-1].split('\n')
    names = [start_token+name for name in names]
```


```python
print ('n samples = ',len(names))

# stride = 1000
for x in names[::1000]:
    print (x)    
```

    n samples =  7944
     Abagael
     Claresta
     Glory
     Liliane
     Prissie
     Geeta
     Giovanne
     Piggy



```python
MAX_LENGTH = max(map(len,names))
print("max length =", MAX_LENGTH)

plt.title('Sequence length distribution')
plt.hist(list(map(len,names)),bins=25);
```

    max length = 16



![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week5/output_5_1.png)


# Text processing

First we need next to collect a "vocabulary" of all unique tokens i.e. unique characters. We can then encode inputs as a sequence of character ids.


```python
len(np.unique([' Abagail',
 ' Abagail',
 ' Abbe',
 ' Abbey',
 ' Abbi',
 ' Abbie',
 ' Abby',
 ' Abigael',
 ' Abigail',
 ' Abigale'])), len(names)
```




    (9, 7944)




```python
#all unique characters go here (not words but characters)
tokens = set(''.join(names))
print(''.join(names)[:10])
tokens = list(tokens)

n_tokens = len(tokens)
print ('n_tokens = ',n_tokens)

assert 50 < n_tokens < 60

```

     Abagael A
    n_tokens =  55



```python
tokens
```




    ['l',
     'i',
     'c',
     'b',
     'm',
     'O',
     'S',
     'd',
     'o',
     'j',
     'q',
     'P',
     ' ',
     'W',
     'Y',
     'p',
     'f',
     'y',
     'h',
     'z',
     '-',
     'g',
     'A',
     'B',
     'R',
     'v',
     'a',
     'J',
     'M',
     'Q',
     'D',
     'e',
     'u',
     'N',
     'C',
     'X',
     "'",
     'k',
     'K',
     'Z',
     'r',
     'x',
     'V',
     'H',
     't',
     'n',
     'E',
     's',
     'L',
     'I',
     'T',
     'w',
     'U',
     'G',
     'F']



### Cast everything from symbols into identifiers

Tensorflow string manipulation is a bit tricky, so we'll work around it. 
We'll feed our recurrent neural network with ids of characters from our dictionary.

To create such dictionary, let's assign 


```python
token_to_id = {char:index for index, char in enumerate(tokens)}
###YOUR CODE HERE: create a dictionary of {symbol -> its  index in tokens }

```


```python
assert len(tokens) == len(token_to_id), "dictionaries must have same size"

for i in range(n_tokens):
    assert token_to_id[tokens[i]] == i, "token identifier must be it's position in tokens list"

print("Seems alright!")
```

    Seems alright!



```python
def to_matrix(names,max_len=None,pad=0,dtype='int32'):
    """Casts a list of names into rnn-digestable matrix"""
    
    max_len = max_len or max(map(len,names)) 
    names_ix = np.zeros([len(names),max_len],dtype) + pad

    for i in range(len(names)):
        name_ix = list(map(token_to_id.get,names[i]))
        # print('name_ix: \n', name_ix)
        names_ix[i,:len(name_ix)] = name_ix

    return names_ix.T
```


```python
from random import sample
to_matrix(names=sample(names, 32))
```




    array([[12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
            12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
           [33, 27, 30, 38, 34, 46, 43, 22, 48, 34, 34, 28, 49, 53, 46, 22,
            23,  6, 22, 28, 30, 24, 43, 30, 54, 33, 30, 33, 24, 30, 27, 46],
           [ 8,  8, 31, 26, 18, 40, 31, 40,  1,  8,  8, 26, 37, 26, 40, 21,
            26, 44, 32, 26, 26,  8,  8, 26, 26, 26, 26,  1, 26, 31,  8,  7],
           [31, 40, 40, 40, 40, 40, 45,  7,  7, 40, 45,  7, 31, 31, 18, 45,
            45, 26, 40, 40, 45, 47, 45, 45, 51, 44, 45, 47, 17, 45,  7,  1],
           [ 0, 21,  1,  0,  1,  8,  7, 17,  1, 40, 26,  7, 17,  0, 26,  8,
             2, 16, 31, 21, 31, 31,  8, 45, 45, 26, 45, 47,  4, 45,  1, 31],
           [17, 31,  2,  1, 47,  0, 31, 44, 26,  1, 45, 17,  0,  0, 40,  0,
            40, 26,  0, 26,  0,  0, 40, 17, 31,  0,  1, 31, 32,  1,  0,  0],
           [45,  0, 37, 31, 44,  0, 40, 18,  0, 45,  0,  0,  0,  0,  7, 26,
             8, 45,  0, 40,  0,  0,  0,  0,  0, 31, 31,  0, 45, 31,  0,  0],
           [ 0,  0,  0,  0,  1,  0, 47,  0,  0, 45,  0,  0,  0,  0,  0,  0,
            16,  1,  0, 31,  0, 26,  0,  0,  0,  0,  0,  0,  7,  0,  0,  0],
           [ 0,  0,  0,  0, 45,  0,  8,  0,  0, 31,  0,  0,  0,  0,  0,  0,
            44,  0,  0, 44,  0,  0,  0,  0,  0,  0,  0,  0,  8,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0, 45,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0, 44,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0, 31,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],
          dtype=int32)




```python
#Example: cast 4 random names to matrices, pad with zeros
print('\n'.join(names[::2000])) 
print(to_matrix(names[::2000]).T) # the first '20' is because there's an indent before each word
```

     Abagael
     Glory
     Prissie
     Giovanne
    [[12 22  3 26 21 26 31  0  0]
     [12 53  0  8 40 17  0  0  0]
     [12 11 40  1 47 47  1 31  0]
     [12 53  1  8 25 26 45 45 31]]


# Recurrent neural network

We can rewrite recurrent neural network as a consecutive application of dense layer to input $x_t$ and previous rnn state $h_t$. This is exactly what we're gonna do now.
<img src="./rnn.png" width=480>

Since we're training a language model, there should also be:
* An embedding layer that converts character id x_t to a vector.
* An output layer that predicts probabilities of next phoneme


```python
import keras
from keras.layers import Concatenate,Dense,Embedding

rnn_num_units = 64
embedding_size = 16

#Let's create layers for our recurrent network
#Note: we create layers but we don't "apply" them yet
embed_x = Embedding(n_tokens,embedding_size) # an embedding layer that converts character ids into embeddings


#a dense layer that maps input and previous state to new hidden state, [x_t,h_t]->h_t+1
get_h_next = Dense(rnn_num_units, activation='relu') ###YOUR CODE HERE

#a dense layer that maps current hidden state to probabilities of characters [h_t+1]->P(x_t+1|h_t+1)
get_probas =  Dense(n_tokens, activation='softmax') ###YOUR CODE HERE 

#Note: please either set the correct activation to Dense or write it manually in rnn_one_step
```


```python
def rnn_one_step(x_t, h_t):
    """
    Recurrent neural network step that produces next state and output
    given prev input and previous state.
    We'll call this method repeatedly to produce the whole sequence.
    
    Follow inline isntructions to complete the function.
    """
    #convert character id into embedding
    x_t_emb = embed_x(tf.reshape(x_t,[-1,1]))[:,0]
    
    #concatenate x embedding and previous h state
    x_and_h = tf.concat([x_t_emb, h_t], axis=1) ###YOUR CODE HERE
    
    #compute next state given x_and_h
    h_next = get_h_next(x_and_h) ###YOUR CODE HERE
    
    #get probabilities for language model P(x_next|h_next)
    output_probas = get_probas(h_next) ###YOUR CODE HERE
    
    return output_probas,h_next
```

### RNN loop

Once rnn_one_step is ready, let's apply it in a loop over name characters to get predictions.

Let's assume that all names are at most length-16 for now, so we can simply iterate over them in a for loop.



```python
MAX_LENGTH
```




    16




```python
input_sequence = tf.placeholder('int32',(MAX_LENGTH,None))
print(input_sequence)
batch_size = tf.shape(input_sequence)[1]

predicted_probas = []
h_prev = tf.zeros([batch_size,rnn_num_units]) #initial hidden state

for t in range(MAX_LENGTH):
    x_t = input_sequence[t]
    probas_next,h_next = rnn_one_step(x_t,h_prev)
    
    h_prev = h_next
    predicted_probas.append(probas_next)
    
predicted_probas = tf.stack(predicted_probas)
```

    Tensor("Placeholder_3:0", shape=(16, ?), dtype=int32)


## RNN: loss and gradients

Let's gather a matrix of predictions for $P(x_{next}|h)$ and the corresponding correct answers.

Our network can then be trained by minimizing crossentropy between predicted probabilities and those answers.


```python
predictions_matrix = tf.reshape(predicted_probas[:-1],[-1,len(tokens)])
answers_matrix = tf.one_hot(tf.reshape(input_sequence[1:],[-1]), n_tokens)
```


```python
from keras.objectives import categorical_crossentropy
loss = tf.reduce_mean(categorical_crossentropy(answers_matrix, predictions_matrix))
optimize = tf.train.AdamOptimizer().minimize(loss)
```

    WARNING:tensorflow:From /home/karen/.local/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:2885: calling reduce_sum (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
    Instructions for updating:
    keep_dims is deprecated, use keepdims instead


### The training loop


```python
from IPython.display import clear_output
from random import sample
s = keras.backend.get_session()
s.run(tf.global_variables_initializer())
history = []
```


```python
input_sequence
```




    <tf.Tensor 'Placeholder_3:0' shape=(16, ?) dtype=int32>




```python
for i in range(1000):
    batch = to_matrix(sample(names,32),max_len=MAX_LENGTH)
    loss_i,_ = s.run([loss,optimize],{input_sequence:batch})
    
    
    history.append(loss_i)
    if (i+1)%100==0:
        clear_output(True)
        plt.plot(history,label='loss')
        plt.legend()
        plt.show()

assert np.mean(history[:10]) > np.mean(history[-10:]), "RNN didn't converge."
```


![png](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/week5/output_28_0.png)


### RNN: sampling
Once we've trained our network a bit, let's get to actually generating stuff. All we need is the `rnn_one_step` function you have written above.


```python
x_t = tf.placeholder('int32',(None,))
h_t = tf.Variable(np.zeros([1,rnn_num_units],'float32'))

next_probs,next_h = rnn_one_step(x_t,h_t)
```


```python
def generate_sample(seed_phrase=' ',max_length=MAX_LENGTH):
    '''
    The function generates text given a phrase of length at least SEQ_LENGTH.
        
    parameters:
        The phrase is set using the variable seed_phrase
        The optional input "N" is used to set the number of characters of text to predict.     
    '''
    x_sequence = [token_to_id[token] for token in seed_phrase]
    s.run(tf.assign(h_t,h_t.initial_value))
    
    #feed the seed phrase, if any
    for ix in x_sequence[:-1]:
        s.run(tf.assign(h_t,next_h),{x_t:[ix]})
    
    #start generating
    for _ in range(max_length-len(seed_phrase)):
        x_probs,_ = s.run([next_probs,tf.assign(h_t,next_h)],{x_t:[x_sequence[-1]]})
        x_sequence.append(np.random.choice(n_tokens,p=x_probs[0]))
    return ''.join([tokens[ix] for ix in x_sequence])
```


```python
for _ in range(10):
    print(generate_sample())
```

     Rltelllllllllll
     Dorinanalllllll
     Dhillelllllllll
     Jiandateenlllll
     Cacbynallllllll
     Gedyreellllllll
     Mouelllllllllll
     Jhrolllllllllll
     Martinellllllll
     Sianlalllllllll



```python
for _ in range(50):
    print(generate_sample(' Trump'))
```

     Trumpolllllllll
     Trumptyllllllll
     Trumpilllllllll
     Trumpelllllllll
     Trumpyltlllllll
     Trumptyllllllll
     Trumpiallllllll
     Trumpelllllllll
     Trumpinllllllll
     Trumponllllllll
     Trumpiellllllll
     Trumpiellllllll
     Trumpiellllllll
     Trumpolllllllll
     Trumpelllllllll
     Trumpalllllllll
     Trumpelllllllll
     Trumpchelllllll
     Trumpolllllllll
     Trumpanllllllll
     Trumpilllllllll
     Trumpilllllllll
     Trumpinnlllllll
     Trumpllllllllll
     Trumpfellllllll
     Trumpllllllllll
     Trumphellllllll
     Trumpetelllllll
     Trumpi-llllllll
     Trumpitllllllll
     Trumpelllllllll
     Trumptiylllllll
     Trumpcellllllll
     Trumpalllllllll
     Trumptlirllllll
     Trumpinllllllll
     Trumpylllllllll
     Trumpelllllllll
     Trumpllllllllll
     Trumpolllllllll
     Trumpeallllllll
     Trumpilllllllll
     Trumpiidlllllll
     Trumphlllllllll
     Trumpalllllllll
     Trumprlllllllll
     Trumpelllllllll
     Trumpelllllllll
     Trumpiallllllll
     Trumpiellllllll


### Try it out!

__Disclaimer:__ This assignment is entirely optional. You won't receive bonus points for it. However, it's a fun thing to do. Please share your results on course forums.

You've just implemented a recurrent language model that can be tasked with generating any kind of sequence, so there's plenty of data you can try it on:

* Novels/poems/songs of your favorite author
* News titles/clickbait titles
* Source code of Linux or Tensorflow
* Molecules in [smiles](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) format
* Melody in notes/chords format
* Ikea catalog titles
* Pokemon names
* Cards from Magic, the Gathering / Hearthstone

If you're willing to give it a try, here's what you wanna look at:
* Current data format is a sequence of lines, so a novel can be formatted as a list of sentences. Alternatively, you can change data preprocessing altogether.
* While some datasets are readily available, others can only be scraped from the web. Try `Selenium` or `Scrapy` for that.
* Make sure MAX_LENGTH is adjusted for longer datasets. There's also a bonus section about dynamic RNNs at the bottom.
* More complex tasks require larger RNN architecture, try more neurons or several layers. It would also require more training iterations.
* Long-term dependencies in music, novels or molecules are better handled with LSTM or GRU

__Good hunting!__

### Bonus level: dynamic RNNs

Apart from keras, there's also a friendly tensorflow API for recurrent neural nets. It's based around the symbolic loop function (aka [scan](https://www.tensorflow.org/api_docs/python/tf/scan)).

This interface allows for dynamic sequence length and comes with some pre-implemented architectures.


```python
class CustomRNN(tf.nn.rnn_cell.BasicRNNCell):
    def call(self,input,state):
        return rnn_one_step(input[:,0],state)
    
    @property
    def output_size(self):
        return n_tokens

cell = CustomRNN(rnn_num_units)

input_sequence = tf.placeholder('int32',(None,None))
    
predicted_probas, last_state = tf.nn.dynamic_rnn(cell,input_sequence[:,:,None],
                                                 time_major=True,dtype='float32')

print(predicted_probas.eval({input_sequence:to_matrix(names[:10],max_len=50)}, session=s).shape)
```

    (50, 10, 55)


Note that we never used MAX_LENGTH in the code above: TF will iterate over however many time-steps you gave it.

You can also use the all the pre-implemented RNN cells:


```python
for obj in dir(tf.nn.rnn_cell)+dir(tf.contrib.rnn):
    if obj.endswith('Cell'):
        print (obj)
```

    BasicLSTMCell
    BasicRNNCell
    GRUCell
    LSTMCell
    MultiRNNCell
    RNNCell
    BasicLSTMCell
    BasicRNNCell
    BidirectionalGridLSTMCell
    Conv1DLSTMCell
    Conv2DLSTMCell
    Conv3DLSTMCell
    ConvLSTMCell
    CoupledInputForgetGateLSTMCell
    FusedRNNCell
    GLSTMCell
    GRUBlockCell
    GRUCell
    GridLSTMCell
    IntersectionRNNCell
    LSTMBlockCell
    LSTMBlockFusedCell
    LSTMCell
    LayerNormBasicLSTMCell
    LayerRNNCell
    MultiRNNCell
    NASCell
    PhasedLSTMCell
    RNNCell
    TimeFreqLSTMCell
    UGRNNCell



```python
input_sequence = tf.placeholder('int32',(None,None))

inputs_embedded = embed_x(input_sequence)

cell = tf.nn.rnn_cell.LSTMCell(rnn_num_units)

state_sequence,last_state = tf.nn.dynamic_rnn(cell,inputs_embedded,dtype='float32')

print('LSTM visible states[time,batch,unit]:', state_sequence)
```

    LSTM visible states[time,batch,unit]: Tensor("rnn_1/transpose_1:0", shape=(?, ?, 64), dtype=float32)

