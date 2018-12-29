
## Notes: Language Representation Models Summary

### Intro

![](../../images/week3/16.png)


回顾过去基于深度学习的 NLP 任务可以发现，几乎绝大多数都比较符合这三层概念。比如很多生成任务的 Seq2Seq 框架中不外乎都有一个 Encoder 和一个 Decoder。对应到这里，__Decoder 更像是一个 Task-specific Model，然后相应的将 Encoder 做一些细微调整，比如引入 Attention 机制等等__

- Eg:
    - 对于一些文本分类任务的结构，则 Encoder 模块与 Task-specific Model 模块的区分更为明显和清晰，Encoder 层可以作为一个相对比较通用的模块来使用, Encoder 负责提取文本特征，最后接上一些全连接层和 Softmax 层便可以当做 Task-specific Model 模块，便完成了一个文本分类任务
    
    

### SVD+LSA -> PLSA -> LDA (introduce prior)

![](../../images/week3/1.png)


![](../../images/week3/2.png)


### Here:

$$PPMI = \log (lift(or: interest)) = \log \frac{p(x, y)}{p(x)p(y)}$$

![](../../images/week3/3.png)



![](../../images/week3/9.png)

![](../../images/week3/10.png)

![](../../images/week3/11.png)

![](../../images/week3/12.png)

![](../../images/week3/13.png)


### Language Model

![](../../images/week3/17.png)

![](../../images/week3/21.png)


### NNLM

![](../../images/week3/18.png)


![](../../images/week3/24.png)






### CBOW


![](../../images/week3/19.png)


![](../../images/week3/26.png)





### SKIP-GRAM


![](../../images/week3/20.png)



![](../../images/week3/25.png)



### Expansion of word2vec

#### PV-DM (CBOW of paragraphs) &  PV-DBOW (Skip-Gram of paragraphs) 


![](../../images/week3/22.png)


### Skip-Thoughts

Skip-thoughts 直接在句子间进行预测，也就是__将 Skip-gram 中以词为基本单位，替换成了以句子为基本单位__，具体做法就是选定一个窗口，遍历其中的句子，然后分别利用当前句子去预测和输出它的上一句和下一句


对于句子的建模利用的 RNN 的 sequence 结构，预测上一个和下一个句子时候，也是利用的一个 sequence 的 RNN 来生成句子中的每一个词，所以这个结构本质上就是一个 Encoder-Decoder 框架，只不过和普通框架不一样的是，Skip-thoughts 有两个 Decoder

- future works:
    - 输入的 Encoder 可以引入 attention 机制, 从而让 Decoder 的输入不再只是依赖 Encoder 最后一个时刻的输出
    - Encoder 和 Decoder 可以利用更深层的结构
    - Decoder 也可以继续扩大，可以预测上下文中更多的句子
    - RNN 也不是唯一的选择，诸如 CNN 以及 2017 年谷歌提出的 Transformer 结构也可以利用进来

- major drawbacks
    - Skip-thoughts 的 Decoder 效率太低
    - 无法在大规模语料上很好的训练
    
    
### Quick-Thoughts

Skip-thoughts 的生成任务改进成为了一个分类任务，具体说来就是把同一个上下文窗口中的句子对标记为正例，把不是出现在同一个上下文窗口中的句子对标记为负例，并将这些句子对输入模型，让模型判断这些句子对是否是同一个上下文窗口中，很明显，这是一个分类任务
    
    
![](../../images/week3/23.png)
















### [GloVe](https://nlp.stanford.edu/pubs/glove.pdf)

![](../../images/week3/4.png)


![](../../images/week3/5.png)


![](../../images/week3/6.png)


![](../../images/week3/7.png)

![](../../images/week3/8.png)

#### GloVe Paper Note


![](../../images/week3/14.png)

![](../../images/week3/15.png)



### CoVe (Contextualized Word Vectors, outputs of the MT-LSTMs)

[Learned in Translation: Contextualized Word Vectors (notes later)] 论文首先用一个 Encoder-Decoder 框架在机器翻译的训练语料上进行预训练，而后用训练好的模型，只取其中的 Embedding 层和 Encoder 层，同时在一个新的任务上设计一个 task-specific 模型，再将原先预训练好的 Embedding 层和 Encoder 层的输出作为这个 task-specific 模型的输入，最终在新的任务场景下进行训练.



#### Encoder

![](../../images/week3/28.png)
![](../../images/week3/29.png)


#### Decoder

![](../../images/week3/30.png)



#### Attention Mechanism

为了决定下一步翻译英语句子中的哪一部分，注意力机制需要从隐向量向前回溯。它使用状态向量来判别每一个隐向量的重要性，为了记录它的观察值，注意力机制会生成一个新的向量，我们可以称之为语境调整状态（context-sdjusted state）





![](../../images/week3/31.png)

然后，生成器会根据语境调整状态来决定要生成哪个单词，接下来语境调整状态会回传到解码器中，让解码器对其翻译的结果有一个准确的感知。解码器一直重复这个过程，直至它完成所有翻译。这就是一个标准的基于注意力机制的编码器-解码器结构，它被用来学习像机器翻译一样的序列到序列任务。
![](../../images/week3/32.png)

当训练过程结束之后，将训练好的 LSTM 提取出来作为编码器用于机器翻译。我们将这个预训练的 LSTM 称作机器翻译 LSTM（MT-LSTM），并使用它生成新句子的隐向量。当我们把这些机器翻译隐向量用于其它的自然语言处理模型时，我们就把它们称作__语境向量(CoVe)__(CoVe 可以被用在任何将向量序列作为输入的模型中)



![](../../images/week3/27.png)






#### CoVe Paper Note

![](../../images/week3/33.png)

![](../../images/week3/34.png)

![](../../images/week3/35.png)


#### CoVe Model Code


```python

```

### Attention

#### Attention notes

- Seq2seq rnn-based model (without attention):

![](../../images/week3/36.jpg)
![](../../images/week3/40.jpg)



- with attention

![](../../images/week3/41.jpg)

在该模型中，定义了一个条件概率：

![](../../images/week3/42.jpg)

其中，$s_i$是decoder中RNN在在i时刻的隐状态

![](../../images/week3/43.jpg)

背景向量ci的计算方式，与传统的Seq2Seq模型直接累加的计算方式不一样，这里的ci是一个权重化（Weighted）之后的值:

![](../../images/week3/44.jpg)

$h_j$ 表示encoder端的第j个词的隐向量，$a_{ij}$表示encoder端的第j个词与decoder端的第i个词之间的权值，表示源端第j个词对目标端第i个词的影响程度

$a_{ij}$的计算公式:

![](../../images/week3/45.jpg)
![](../../images/week3/52.png)


$e_{ij}$ 表示一个对齐模型，用于衡量encoder端的位置j个词，对于decoder端的位置i个词的对齐程度（影响程度）(i.e. decoder端生成位置i的词时，有多少程度受encoder端的位置j的词影响)

对齐模型eij的计算方式有很多种，不同的计算方式，代表不同的Attention模型，最简单且最常用的的对齐模型是dot product乘积矩阵，即把target端的输出隐状态ht与source端的输出隐状态进行矩阵乘

![](../../images/week3/46.jpg)


![](../../images/week3/37.jpg)


![](../../images/week3/38.jpg)


权重 $\alpha$ 是怎么来的呢？常见有三种方法：


- $\alpha_{0}^1=cos\_sim(z_0, h_1)$
- $\alpha_0 =neural\_network(z_0, h)$
- $\alpha_0 = h^TWz_0$


思想就是根据当前解码“状态”判断输入序列的权重分布


attention其实是以下的机制:

![](../../images/week3/39.jpg)

模型通过Q和K的匹配计算出权重，再结合V得到输出：

$$Attention(Q, K, V) = softmax(sim(Q, K))V$$


- 模型分类
    - Soft/Hard Attention
        - soft attention：传统attention，可被嵌入到模型中去进行训练并传播梯度
        - hard attention：不计算所有输出，依据概率对encoder的输出采样，在反向传播时需采用蒙特卡洛进行梯度估计
    - Global/Local Attention
        - global attention：传统attention，对所有encoder输出进行计算
            - 传统的Attention model一样。所有的hidden state都被用于计算Context vector 的权重，即变长的对齐向量at，其长度等于encoder端输入句子的长度
            ![](../../images/week3/47.jpg)
            在t时刻，首先基于decoder的隐状态ht和源端的隐状态hs，计算一个变长的隐对齐权值向量$a_t$
            ![](../../images/week3/48.jpg)
            得到对齐向量$a_t$之后，就可以通过加权平均的方式，得到上下文向量$c_t$
        - local attention：介于soft和hard之间，会预测一个位置并选取一个窗口进行计算
            - ![](../../images/week3/49.jpg)
            - Local Attention首先会为decoder端当前的词，预测一个source端对齐位置（aligned position）$p_t$，然后基于$p_t$选择一个窗口，用于计算背景向量$c_t$
            ![](../../images/week3/50.jpg)
            - S是encoder端句子长度，vp和wp是模型参数, 此时，对齐向量at的计算公式:
            ![](../../images/week3/51.jpg)
    - Self Attention
        - ![](../../images/week3/50.png)
        - Self-attention 中的multiple-heads mechanism便是将这样的操作分别进行多次，让句子的表征充分学习到不同的侧重点，最终将这些多头学习出来的表征 concat 到一起，然后再同一个全连接网络，便可以得到这个句子最终 Self-attention 下新的表示, 其中的每一个头的操作过程用公式表示如下，需要注意的是 softmax 是针对矩阵的 row 方向进行操作得到的。所以，说白了，这个公式表示的意思就是针对 V 进行加权求和，加权权值通过 Q 和 K 的点乘得到:
        - ![](../../images/week3/51.png)
        - 这里给出一例，下图只是两个 head 学习到的交融模式，如果多达 16 个 head，这样的交融模式还要重复16次 (而相应的在 ELMo 与 GPT 中，它们并没有用上这种交融模式，也就是它们本质上还是一个单向的模型，ELMo 稍微好一点，将两个单向模型的信息 concat起 来。GPT 则只用了单向模型，这是因为它没有用上 Transformer Encoder、只用了 Decdoer 的天生基因决定的):
        - ![](../../images/week3/55.png)
        - 传统attention是计算Q和K之间的依赖关系，__而self attention则分别计算Q和K自身的依赖关系__
        - Self Attention 分别在source端和target端进行，仅与source input或者target input自身相关的Self Attention，捕捉source端或target端自身的词与词之间的依赖关系；然后再把source端的得到的self Attention加入到target端得到的Attention中，捕捉source端和target端词与词之间的依赖关系
        - Self Attention 的具体计算方式如图所示:
        ![](../../images/week3/52.jpg)
       - 从All Attention的结构示意图可以发现，Encoder和decoder是层叠多了类似的__Multi-Head Attention__单元构成，而每一个Multi-Head Attention单元由多个结构相似的__Scaled Dot-Product Attention__单元组成
        ![](../../images/week3/53.jpg)
        - Self Attention也是在Scaled Dot-Product Attention单元里面实现的
            - 首先把输入Input经过线性变换分别得到Q、K、V
            - 然后把Q和K做dot Product相乘，得到输入Input词与词之间的依赖关系
            - 然后经过尺度变换（scale）、掩码（mask）和softmax操作，得到最终的Self Attention矩阵 (尺度变换是为了防止输入值过大导致训练不稳定，mask则是为了保证时间的先后关系)
            - 最后，把encoder端self Attention计算的结果加入到decoder做为k和V，结合decoder自身的输出做为q，得到encoder端的attention与decoder端attention之间的依赖关系
    - Other Attention
        - __Hierarchical Attention__构建了两个层次的Attention Mechanism，第一个层次是对句子中每个词的attention，即word attention；第二个层次是针对文档中每个句子的attention，即sentence attention
         ![](../../images/week3/54.jpg)
        - __Attention over Attention__
            - 两个输入，一个Document和一个Query，分别用一个双向的RNN进行特征抽取，得到各自的隐状态h（doc）和h（query)
            - 然后基于query和doc的隐状态进行dot product，得到query和doc的attention关联矩阵
            - 然后按列（column）方向进行softmax操作，得到query-to-document的attention 值a（t）
            - 按照行（row）方向进行softmax操作，得到document-to-query的attention值b（t）
            - 再进行attention操作，即attention over attention得到最终query与document的关联矩阵




#### Transformer

OpenAI Transformer是一类可迁移到多种NLP任务的，基于Transformer的语言模型。它的基本思想同ULMFiT相同，都是在尽量不改变模型结构的情况下__将预训练的语言模型应用到各种任务__。不同的是，OpenAI Transformer主张用Transformer结构，而ULMFiT中使用的是基于RNN的语言模型


![](../../images/week3/64.jpg)



Besides _Self Attention_, 在 Transformer 的 Encoder 中，还有一些其他设计，比如:
- 加入 position embedding（因为 Transformer 的 Encoder 中不是时序输入词序列，因此 __position embedding 也是主要位置信息__);
- Residual 结构，使得模型的训练过程更为平稳;
- normalization 层
- feed forward 层（本质上是一个两层的全连接网络，中间加一个 ReLu 的激活函数）

Decoder 的结构与此类似，只不过在进行 decode 的时候，会__将 Encoder 这边的输出作为 Decoder 中 Self-attention 时的 K 和 V__

![](../../images/week3/53.png)


对于 decode 过程，具体来看，大致过程如下:

![](../../images/week3/54.png)

![](../../images/week3/1.gif)






#### Seq2Seq Attention TF source code (from scratch later)

__tf.nn.seq2seq__文件共实现了5个seq2seq函数

- __1. basic_rnn_seq2seq__: 最简单版本
    - _输入和输出都是embedding的形式_;
    - 最后一步的state vector作为decoder的initial state;
    - encoder和decoder用相同的RNN cell， 但不共享权值参数
- __2. tied_rnn_seq2seq__: 同1，但是encoder和decoder共享权值参数
- __3. embedding_rnn_seq2seq__: 同1，但输入和输出改为id的形式，函数会在内部创建分别用于encoder和decoder的embedding matrix
- __4. embedding_tied_rnn_seq2seq__: 同2，但输入和输出改为id形式，函数会在内部创建分别用于encoder和decoder的embedding matrix
- __5. embedding_attention_seq2seq__: 同3，但多了attention机制




#### tf.nn.seq2seq.embedding_attention_seq2seq


```python
# T代表time_steps, 时序长度
def embedding_attention_seq2seq(encoder_inputs,  # [T, batch_size], int32 id tensor list
                                decoder_inputs,  # [T, batch_size], int32 id tensor list
                                cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1,      # attention的抽头数量
                                output_projection=None, #decoder的投影矩阵
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
```

Encoder

- 创建了一个embedding matrix
- 计算encoder的output和state
- 生成attention states，用于计算attention




```python
encoder_cell = rnn_cell.EmbeddingWrapper(      
                                        cell, 
                                        embedding_classes=num_encoder_symbols, # 编码的符号数，即词表大小
                                        embedding_size=embedding_size) # 词向量的维度

encoder_outputs, encoder_state = rnn.rnn(
                                        encoder_cell, 
                                        encoder_inputs, 
                                        dtype=dtype) #  [T，batch_size，size]

top_states = [array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs]    # T * [batch_size, 1, size]

attention_states = array_ops.concat(1, top_states) # [batch_size,T,size]
```

上面的EmbeddingWrapper, 是RNNCell的前面加一层embedding，作为encoder_cell, input就可以是word的id


```python
class EmbeddingWrapper(RNNCell):
    def __init__(self, cell, embedding_classes, embedding_size, initializer=None):
    def __call__(self, inputs, state, scope=None):
      #生成embedding矩阵[embedding_classes,embedding_size]
      #inputs: [batch_size, 1]
      #return : (output, state)
```

Decoder

- 生成decoder的cell，通过OutputProjectionWrapper类对输入参数中的cell实例包装实现


```python
# Decoder.
    output_size = None
    if output_projection is None:
      cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
      output_size = num_decoder_symbols
    if isinstance(feed_previous, bool):
      return embedding_attention_decoder(
          ...
      )
```

上面的OutputProjectionWrapper将输出映射成想要的维度


```python
class OutputProjectionWrapper(RNNCell):
    def __init__(self, cell, output_size): # output_size:映射后的size
    def __call__(self, inputs, state, scope=None):
      #init 返回一个带output projection的 rnn_cell
```


```python
def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
    # 第一步创建了解码用的embedding
    embedding = variable_scope.get_variable("embedding",
                                            [num_symbols, embedding_size])

    # 第二步创建了一个循环函数loop_function，用于将上一步的输出映射到词表空间，输出一个word embedding作为下一步的输入
    loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None
    emb_inp = [
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
    # T * [batch_size, embedding_size]
    return attention_decoder(
        emb_inp,
        initial_state,
        attention_states,
        cell,
        output_size=output_size,
        num_heads=num_heads,
        loop_function=loop_function,
        initial_state_attention=initial_state_attention)
```


```python
def attention_decoder(decoder_inputs,    #T * [batch_size, input_size]
                      initial_state,     #[batch_size, cell.states]
                      attention_states,  #[batch_size, attn_length , attn_size]
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
```

__num_heads__:

![](../../images/week3/37.png)

attention就是对信息的加权求和，一个attention head对应了一种加权求和方式，这个参数定义了用多少个attention head去加权求和，所以公式三可以进一步表述为$\sum^{num\_heads}_{j=1}\sum^{T_{A}}_{i=1}a_{i,j}h_{i}$

- $W_{1}*h_{i}$用的是卷积的方式实现，返回的tensor的形状是[batch_size, attn_length, 1, attention_vec_size]




```python
# To calculate W1 * h_t we use a 1-by-1 convolution
hidden = array_ops.reshape(attention_states, 
                           [-1, attn_length, 1, attn_size])
hidden_features = []
v = []
attention_vec_size = attn_size  # Size of query vectors for attention.

for a in xrange(num_heads):
    k = variable_scope.get_variable("AttnW_%d" % a,
                                    [1, 1, attn_size, attention_vec_size])
    hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
    v.append(
          variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))
```

- $W_{2}*d_{t}$，此项是通过下面的线性映射函数linear实现:


```python
for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_%d" % a):
          # query对应当前隐层状态d_t
          y = linear(query, attention_vec_size, True)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
          # 计算u_t
          s = math_ops.reduce_sum(
              v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
          a = nn_ops.softmax(s)
          # 计算 attention-weighted vector d.
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
              [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size]))
```

### ELMo

#### ELMo notes

Goal: 一个预训练的词表示应该能够包含丰富的句法和语义信息，并且能够对多义词进行建模


![](../../images/week3/55.jpg)


ELMo 利用语言模型来获得一个上下文相关的预训练表示:

![](../../images/week3/38.png)

基本框架是一个双层的 Bi-LSTM，不过在第一层和第二层之间加入了一个残差结构（一般来说，残差结构能让训练过程更稳定)

在 ELMo 中使用的是一个双向的 LSTM 语言模型，由一个前向和一个后向语言模型构成，目标函数就是取这两个方向语言模型的最大似然. ELMo 的__基本框架是 2-stacked biLSTM + Residual 的结构__, ELMo 的训练目标函数为:

![](../../images/week3/39.png)


![](../../images/week3/56.jpg)


在预训练好这个语言模型之后，ELMo 就是根据下面的公式来用作词表示，其实就是把这个双向语言模型的每一中间层进行一个求和:

![](../../images/week3/57.jpg)

__总结一下，不像传统的词向量，每一个词只对应一个词向量，ELMo 利用预训练好的双向语言模型，然后根据具体输入从该语言模型中可以得到上下文依赖的当前词表示（对于不同上下文的同一个词的表示是不一样的），再当成特征加入到具体的 NLP 有监督模型里__

不过和普通 RNN 结构的不同之处在于，其主要改进在于输入层和输出层不再是 word，而是变为了一个 char-based CNN 结构，ELMo 在输入层和输出层考虑了使用同样的这种结构，该结构如下图示:

![](../../images/week3/40.png)


_$Note^*$:_

输入层和输出层都使用了 CNN 结构


在 CBOW 中的普通 Softmax 方法中，为了计算每个词的概率大小，使用的如下公式的计算方法:

![](../../images/week3/41.png)

现在我们假定 char-based CNN 模型是现成已有的，对于任意一个目标词都可以得到一个向量表示 CNN(tk) ，当前时刻的 LSTM 的输出向量为 h，那么便可以通过同样的方法得到目标词的概率大小:

![](../../images/week3/42.png)


这种先经过 CNN 得到词向量，然后再计算 Softmax 的方法叫做 CNN Softmax

利用 CNN 解决有三点优势值得注意:

-  CNN 能减少普通做 Softmax 时全连接层中的必须要有的 $|V|*h$ 的参数规模，只需保持 CNN 内部的参数大小即可。一般来说，CNN 中的参数规模都要比 $|V|*h$ 的参数规模小得多 (CNN Softmax 的好处就在于能够做到对于不同的词，映射参数都是共享的，这个共享便体现在使用的 CNN 中的参数都是同一套，从而大大减少参数的规模)

- CNN 可以解决 OOV （Out-of-Vocabulary）问题，这个在翻译问题中尤其头疼

- 在预测阶段，CNN 对于每一个词向量的计算可以预先做好，更能够减轻 inference 阶段的计算压力



最终 ELMo 的主要结构便如下图（b）所示，可见输入层和输出层都是一个 CNN，中间使用 Bi-LSTM 框架:


![](../../images/week3/43.png)

$s_j$ 便是针对每一层的输出向量，利用一个 softmax 的参数来学习不同层的权值参数，因为不同任务需要的词语意义粒度也不一致，一般认为浅层的表征比较倾向于句法，而高层输出的向量比较倾向于语义信息。因此通过一个 softmax 的结构让任务自动去学习各层之间的权重

![](../../images/week3/44.png)


![](../../images/week3/45.png)




#### ELMo source code and usage

- 将ELMo向量  $ELMo_k^{task}$ 与传统的词向量  $x_{k}$ 拼接成  $[x_{k};ELMo_k^{task}]$  后，输入到对应具体任务的RNN中。
- 将ELMo向量放到模型输出部分，与具体任务RNN输出的  $h_{k}$ 拼接成 $[h_{k};ELMo_k^{task}]$



#### [tf src code](https://github.com/allenai/bilm-tf/tree/master/bilm)

#### Directly use by import from TF-Hub


```python
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import keras.layers as layers
from keras.models import Model

# Initialize session
sess = tf.Session()
K.set_session(sess)

# Instantiate the elmo model
elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())


def ElmoEmbedding(x):
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)), 
                      signature="default", 
                      as_dict=True)["default"]

input_text = layers.Input(shape=(1,), dtype=tf.string)
embedding = layers.Lambda(ElmoEmbedding, output_shape=(1024,))(input_text)
dense = layers.Dense(256, activation='relu')(embedding)
pred = layers.Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[input_text], outputs=pred)

model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
model.summary()
```

    INFO:tensorflow:Using /tmp/tfhub_modules to cache modules.
    INFO:tensorflow:Downloading TF-Hub Module 'https://tfhub.dev/google/elmo/1'.



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-2-850bfcc5f106> in <module>
         10 
         11 # Instantiate the elmo model
    ---> 12 elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
         13 sess.run(tf.global_variables_initializer())
         14 sess.run(tf.tables_initializer())


    /usr/local/lib/python3.6/dist-packages/tensorflow_hub/module.py in __init__(self, spec, trainable, name, tags)
        103     """
        104     self._graph = tf.get_default_graph()
    --> 105     self._spec = as_module_spec(spec)
        106     self._trainable = trainable
        107 


    /usr/local/lib/python3.6/dist-packages/tensorflow_hub/module.py in as_module_spec(spec)
         29     return spec
         30   elif isinstance(spec, str):
    ---> 31     return native_module.load_module_spec(spec)
         32   else:
         33     raise ValueError("Unknown module spec type: %r" % type(spec))


    /usr/local/lib/python3.6/dist-packages/tensorflow_hub/native_module.py in load_module_spec(path)
         97     tf.OpError: on file handling exceptions.
         98   """
    ---> 99   path = compressed_module_resolver.get_default().get_module_path(path)
        100   module_def_path = _get_module_proto_path(path)
        101   module_def_proto = module_def_pb2.ModuleDef()


    /usr/local/lib/python3.6/dist-packages/tensorflow_hub/resolver.py in get_module_path(self, handle)
        383     """
        384     if self.is_supported(handle):
    --> 385       return self._get_module_path(handle)
        386     else:
        387       raise UnsupportedHandleError(


    /usr/local/lib/python3.6/dist-packages/tensorflow_hub/resolver.py in _get_module_path(self, handle)
        465       raise UnsupportedHandleError(
        466           self._create_unsupported_handle_error_msg(handle))
    --> 467     return resolver.get_module_path(handle)
        468 
        469   def _create_unsupported_handle_error_msg(self, handle):


    /usr/local/lib/python3.6/dist-packages/tensorflow_hub/resolver.py in get_module_path(self, handle)
        383     """
        384     if self.is_supported(handle):
    --> 385       return self._get_module_path(handle)
        386     else:
        387       raise UnsupportedHandleError(


    /usr/local/lib/python3.6/dist-packages/tensorflow_hub/compressed_module_resolver.py in _get_module_path(self, handle)
        103 
        104     return resolver.atomic_download(handle, download, module_dir,
    --> 105                                     self._lock_file_timeout_sec())
        106 
        107   def _lock_file_timeout_sec(self):


    /usr/local/lib/python3.6/dist-packages/tensorflow_hub/resolver.py in atomic_download(handle, download_fn, module_dir, lock_file_timeout_sec)
        311     tf.logging.info("Downloading TF-Hub Module '%s'.", handle)
        312     tf.gfile.MakeDirs(tmp_dir)
    --> 313     download_fn(handle, tmp_dir)
        314     # Write module descriptor to capture information about which module was
        315     # downloaded by whom and when. The file stored at the same level as a


    /usr/local/lib/python3.6/dist-packages/tensorflow_hub/compressed_module_resolver.py in download(handle, tmp_dir)
         99 
        100       url_opener = url.build_opener(LoggingHTTPRedirectHandler)
    --> 101       response = url_opener.open(request)
        102       return resolver.download_and_uncompress(cur_url, response, tmp_dir)
        103 


    /usr/lib/python3.6/urllib/request.py in open(self, fullurl, data, timeout)
        524             req = meth(req)
        525 
    --> 526         response = self._open(req, data)
        527 
        528         # post-process response


    /usr/lib/python3.6/urllib/request.py in _open(self, req, data)
        542         protocol = req.type
        543         result = self._call_chain(self.handle_open, protocol, protocol +
    --> 544                                   '_open', req)
        545         if result:
        546             return result


    /usr/lib/python3.6/urllib/request.py in _call_chain(self, chain, kind, meth_name, *args)
        502         for handler in handlers:
        503             func = getattr(handler, meth_name)
    --> 504             result = func(*args)
        505             if result is not None:
        506                 return result


    /usr/lib/python3.6/urllib/request.py in https_open(self, req)
       1359         def https_open(self, req):
       1360             return self.do_open(http.client.HTTPSConnection, req,
    -> 1361                 context=self._context, check_hostname=self._check_hostname)
       1362 
       1363         https_request = AbstractHTTPHandler.do_request_


    /usr/lib/python3.6/urllib/request.py in do_open(self, http_class, req, **http_conn_args)
       1316             try:
       1317                 h.request(req.get_method(), req.selector, req.data, headers,
    -> 1318                           encode_chunked=req.has_header('Transfer-encoding'))
       1319             except OSError as err: # timeout error
       1320                 raise URLError(err)


    /usr/lib/python3.6/http/client.py in request(self, method, url, body, headers, encode_chunked)
       1237                 encode_chunked=False):
       1238         """Send a complete request to the server."""
    -> 1239         self._send_request(method, url, body, headers, encode_chunked)
       1240 
       1241     def _send_request(self, method, url, body, headers, encode_chunked):


    /usr/lib/python3.6/http/client.py in _send_request(self, method, url, body, headers, encode_chunked)
       1283             # default charset of iso-8859-1.
       1284             body = _encode(body, 'body')
    -> 1285         self.endheaders(body, encode_chunked=encode_chunked)
       1286 
       1287     def getresponse(self):


    /usr/lib/python3.6/http/client.py in endheaders(self, message_body, encode_chunked)
       1232         else:
       1233             raise CannotSendHeader()
    -> 1234         self._send_output(message_body, encode_chunked=encode_chunked)
       1235 
       1236     def request(self, method, url, body=None, headers={}, *,


    /usr/lib/python3.6/http/client.py in _send_output(self, message_body, encode_chunked)
       1024         msg = b"\r\n".join(self._buffer)
       1025         del self._buffer[:]
    -> 1026         self.send(msg)
       1027 
       1028         if message_body is not None:


    /usr/lib/python3.6/http/client.py in send(self, data)
        962         if self.sock is None:
        963             if self.auto_open:
    --> 964                 self.connect()
        965             else:
        966                 raise NotConnected()


    /usr/lib/python3.6/http/client.py in connect(self)
       1390             "Connect to a host on a given (SSL) port."
       1391 
    -> 1392             super().connect()
       1393 
       1394             if self._tunnel_host:


    /usr/lib/python3.6/http/client.py in connect(self)
        934         """Connect to the host and port specified in __init__."""
        935         self.sock = self._create_connection(
    --> 936             (self.host,self.port), self.timeout, self.source_address)
        937         self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        938 


    /usr/lib/python3.6/socket.py in create_connection(address, timeout, source_address)
        711             if source_address:
        712                 sock.bind(source_address)
    --> 713             sock.connect(sa)
        714             # Break explicitly a reference cycle
        715             err = None


    KeyboardInterrupt: 


### ULMFiT

ULMFiT是一种有效的NLP迁移学习方法，核心思想是通过精调预训练的语言模型完成其他NLP任务

ULMFiT的过程分为三步:


![](../../images/week3/47.png)
![](../../images/week3/63.jpg)


- ASGD(Averaged SGD):
    - 是指先将模型训练到一定 epoch，然后再将其后的每一轮权值进行平均后，得到最终的权值
    - 普通的 SGD 方法权值更新过程为：
        - ![](../../images/week3/48.png)
    - ASGD 则把它变成了:
        - ![](../../images/week3/49.png)
        - 其中 T 是一个阈值，而 K 是总共的迭代次数，把model迭代到第 T 次之后，对该参数在其后的第 T 轮到最后一轮之间的所有值求平均，


- 两种fine-tuning方法：
    - Discriminative fine-tuning
        - 因为网络中不同层可以捕获不同类型的信息，因此在精调时也应该使用不同的learning rate。作者为每一层赋予一个学习率  $\eta^{l}$ ，实验后发现，首先通过精调模型的最后一层L确定学习率  $\eta^{L}$ ，再递推地选择上一层学习率进行精调的效果最好，递推公式为:  $\eta^{l-1} =\frac{ \eta^{l}}{2.6}$
    - Slanted triangular learning rates (STLR)
        - 为了针对特定任务选择参数，理想情况下需要在训练开始时让参数快速收敛到一个合适的区域，之后进行精调。为了达到这种效果，作者提出STLR方法，即让LR在训练初期短暂递增，在之后下降。如上图的右上角所示
        - ![](../../images/week3/46.png)
        - parameters:
            - T: number of training iterations
            - cut_frac: fraction of iterations we increase the LR
            - cut: the iteration when we switch from increasing to decreasing the LR
            - p: the fraction of the number of iterations we have increased or will decrease the LR respectively
            - ratio: specifies how much smaller the lowest LR $\eta_{min}$ is from the max LR  $\eta_{max}$
            - $\eta_{t}$  : the LR at iteration t
            - $Note^*:$ in the paper, $cut\_frac=1$, ration=32, $\eta_{max} = 0.01$
            
            
####  Target task classifier fine-tuning


为了完成分类任务的精调，作者在最后一层添加了两个线性block:
- 每个都有batch-norm和dropout
- 使用ReLU作为中间层激活函数
- 最后经过softmax输出分类的概率分布



Details:

- Concat pooling
    - 第一个线性层的输入是最后一个隐层状态的池化。因为文本分类的关键信息可能在文本的任何地方，所以只是用最后时间步的输出是不够的。作者将最后时间步   $h_{T}$ 与尽可能多的时间步 $H= {h_{1},... , h_{T}}$ 池化后拼接起来，以  $h_{c} = [h_{T}, maxpool(H), meanpool(H)]$ 作为输入
    
- Gradual unfreezing
    - 由于过度精调会导致模型遗忘之前预训练得到的信息，作者提出逐渐unfreeze网络层的方法，从最后一层开始unfreeze和精调，由后向前地unfreeze并精调所有层
    
- BPTT for Text Classification (BPT3C) 
    - 为了在large documents上进行模型精调，作者将文档分为固定长度为b的batches，并在每个batch训练时记录mean和max池化，梯度会被反向传播到对最终预测有贡献的batches
    
- Bidirectional language model
    - 在paper中，分别独立地对前向和后向LM做了精调，并将两者的预测结果平均。两者结合后结果有0.5-0.7的提升。
    
    
    

    



            
 
 

### OpenAI GPT

#### GPT notes

2018 年早些时候谷歌的 Generating Wikipedia by Summarizing Long Sequences，GPT 名称中的 Generative 便是源自这篇文章，二者都有用到生成式方法来训练模型，也就是生成式 Decoder。



训练过程分为两步：

- 1. Unsupervised pre-training

__主要亮点在于利用了Transformer网络代替了LSTM作为语言模型来更好的捕获长距离语言结构__


![](../../images/week3/58.jpg)


在具体 NLP 任务有监督微调时，与 ELMo 当成特征的做法不同，OpenAI GPT 不需要再重新对任务构建新的模型结构，而是直接在 Transformer 这个语言模型上的最后一层接上 softmax 作为任务输出层，然后再对这整个模型进行微调

![](../../images/week3/59.jpg)

- 2. Supervised fine-tuning


有了预训练的语言模型之后，对于有标签的训练集 $C$ ，给定输入序列 $x^{1}, ..., x^{m}$ 和标签 $y$，可以通过语言模型得到  $h_{l}^{m}$ ，经过输出层后对  $y$  进行预测


$$p(y|x^1, x^2, ..., x^m) = softmax(h_l^m W_y)$$


$$L_2(C) = \sum_{x,y} \log p(y|x^1, x^2, ..., x^m)$$


整个任务的目标函数为:

$$L_3(C) = L_2(C)+ \lambda L_1(C)$$

其中 $L_2$ 是 task-specific 的目标函数， $L_1$ 则是语言模型的目标函数

Result:

- 计算速度比循环神经网络更快，易于并行化
- 实验结果显示Transformer的效果比ELMo和LSTM网络更好


Summary:


从Wrod Embedding到OpenAI Transformer，NLP中的迁移学习__从最初使用word2vec、GLoVe进行字词的向量表示，到ELMo可以提供前几层的权重共享，再到ULMFiT和OpenAI Transformer的整个预训练模型的精调__，大大提高了NLP基本任务的效果。同时，多项研究也表明，以语言模型作为预训练模型，不仅可以捕捉到文字间的语法信息，更可以捕捉到语义信息，为后续的网络层提供高层次的抽象信息。另外，基于Transformer的模型在一些方面也展现出了优于RNN模型的效果。




#### GPT Model Code


```python
# source code: github openAI finetune-transformer-lm/train.py

# DROPOUT
def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 
                          keep_prob=1-pdrop)
    return x

# EMBEDDING
def embed(X, we):
    we = convert_gradient_to_tensor(we)
    e = tf.gather(we, X)
    h = tf.reduce_sum(e, 2)
    return h

# CONV1D
def conv1d(x, scope, nf, rf, 
           w_init=tf.random_normal_initializer(stddev=0.02), 
           b_init=tf.constant_initializer(0), 
           pad='VALID', 
           train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        if rf == 1: #faster 1x1 conv
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf]) # c = xw+b
        else: #was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c

# FEED FORWARD NN
def mlp(x, scope, n_state, train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        act = act_fns[afn]
        h = act(conv1d(x, 'c_fc', n_state, 1, train=train))
        h2 = conv1d(h, 'c_proj', nx, 1, train=train)
        h2 = dropout(h2, resid_pdrop, train)
        return h2
    
# MULTI-HEAD SELF ATTENTION 
## generate Attention weights
def mask_attn_weights(w):
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)
    return w
## Attention Helper
### Archtecture: matmul(dropout(softmax(masked(scale(matmul(Q,K))))), v), as in the self attention diagram 1 above in the attention section
def _attn(q, k, v, train=False, scale=False):
    w = tf.matmul(q, k)

    if scale:
        n_state = shape_list(v)[-1]
        w = w*tf.rsqrt(tf.cast(n_state, tf.float32))

    w = mask_attn_weights(w)
    w = tf.nn.softmax(w)

    w = dropout(w, attn_pdrop, train)

    a = tf.matmul(w, v)
    return a

# Attention
### Archtecture: matmul(dropout(softmax(masked(scale(matmul(Q,K))))), v), as in the self attention diagram 2 above in the attention section
def split_heads(x, n, k=False):
    if k:
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])
    
def attn(x, scope, n_state, n_head, train=False, scale=False):
    assert n_state%n_head==0
    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3, 1, train=train) # faster 1x1 conv
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        a = _attn(q, k, v, train=train, scale=scale)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, 1, train=train)
        a = dropout(a, resid_pdrop, train)
        return a

# BLOCK
# Architecture: masked multi self-attention --> layer norm --> feed forward --> layer norm --> output
def block(x, scope, train=False, scale=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(x, 'attn', nx, n_head, train=train, scale=scale)
        n = norm(x+a, 'ln_1')
        m = mlp(n, 'mlp', nx*4, train=train)
        h = norm(n+m, 'ln_2')
        return h
    
    
    
# MODEL
## Logits
def clf(x, ny, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), train=False):
    with tf.variable_scope('clf'):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w)+b
    
def model(X, M, Y, train=False, reuse=False):
    with tf.variable_scope('model', reuse=reuse):
        # n_special=3，作者把数据集分为三份
        # n_ctx 应该是 n_context
        we = tf.get_variable(name="we", 
                             shape=[n_vocab+n_special+n_ctx, n_embd], 
                             initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, embd_pdrop, train)

        X = tf.reshape(X, [-1, n_ctx, 2]) # [batch_size, n_context, 2]
        M = tf.reshape(M, [-1, n_ctx])    # [batch_size, n_context]

        # 1. Embedding
        h = embed(X, we)

        # 2. transformer block
        for layer in range(n_layer):
            h = block(h, 'h%d'%layer, train=train, scale=True)

        # 3. language model loss
        lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
        lm_logits = tf.matmul(lm_h, we, transpose_b=True)
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, 
                                                                   labels=tf.reshape(X[:, 1:, 0], [-1]))
        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])
        lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)

        # 4. classifier loss
        clf_h = tf.reshape(h, [-1, n_embd])
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), 
                                             tf.float32), 
                                     1), 
                           tf.int32)
        clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32)*n_ctx+pool_idx)

        clf_h = tf.reshape(clf_h, [-1, 2, n_embd])
        if train and clf_pdrop > 0:
            shape = shape_list(clf_h)
            shape[1] = 1
            clf_h = tf.nn.dropout(clf_h, 1-clf_pdrop, shape)
        clf_h = tf.reshape(clf_h, [-1, n_embd])
        clf_logits = clf(clf_h, 1, train=train)
        clf_logits = tf.reshape(clf_logits, [-1, 2])

        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)
        return clf_logits, clf_losses, lm_losses
```

### BERT

#### Bert notes



对比ELMo，虽然都是“双向”，但目标函数其实是不同的:

- ELMo是分别以$P(w_i| w_1, ...w_{i-1})$ 和 $P(w_i|w_{i+1}, ...w_n)$ 作为目标函数，独立训练处两个representation然后拼接
- BERT则是以 $P(w_i|w_1,  ...,w_{i-1}, w_{i+1},...,w_n)$ 作为目标函数训练LM

这篇论文把预训练语言表示方法分为了基于特征的方法（代表 ELMo）和基于微调的方法（代表 OpenAI GPT）。而目前这两种方法在预训练时都是使用单向的语言模型来学习语言表示


![](../../images/week3/60.jpg)


这篇论文证明了使用双向的预训练效果更好。其实这篇论文方法的整体框架和 GPT 类似，是进一步的发展。具体的，BERT 是使用 Transformer 的编码器来作为语言模型，在语言模型预训练的时候，提出了两个新的目标任务（即masked语言模型 MLM 和预测下一个句子的任务)


#### Method

在语言模型上，BERT 使用的是 Transformer 编码器，并且设计了一个小一点的 base 结构和一个更大的网络结构

![](../../images/week3/61.jpg)

对比一下三种语言模型结构:

- BERT 使用的是 Transformer 编码器，由于 self-attention 机制，所以模型上下层直接全部互相连接的。
- OpenAI GPT 使用的是 Transformer 编码器，它是一个需要__从左到右的受限制__的 Transformer
- ELMo 使用的是双向 LSTM，虽然是双向的，但是也只是在两个单向的 LSTM 的最高层进行简单的拼接。


__所以只有 BERT 是真正在模型所有层中是双向的__

![](../../images/week3/62.jpg)


$Note^*$:

- 而在模型的输入方面，BERT 做了更多的细节，如:
    - 使用了 __WordPiece embedding__ 作为词向量
    - 加入了位置向量和句子切分向量。此外，作者还在每一个文本输入前加入了一个 CLS 向量，后面会有这个向量作为具体的分类向量
    
#### Masked-LM
然而，使用双向 Transformer 会有一个问题。正如上面的分析，即便对于 Base 版 BERT 来说，经过 12 个 block，每个 block 内部都有 12 个多头注意力机制，到最后一层的输出，序列中每个位置上对应的词向量信息，早已融合了输入序列中所有词的信息。

而普通的语言模型中，是通过某个词的上下文语境预测当前词的概率。如果直接把这个套用到 Transformer 的 Encoder 中，会发现待预测的输出和序列输入已经糅合在一块了，说白了就是 Encoder 的输入已经包含了正确的监督信息了，相当于给模型泄题了，如此__普通语言模型的目标函数无法直接套用__。

> 那么，如何解决 Self-attention 中带来了表征性能卓越的双向机制，却又同时带来信息泄露的这一问题？ 

BERT 的作者很快联想到了，如果把原来要预测整个句子的输出，改为只预测这个句子中的某个词，并且把输入中这个词所在位置挖空, 输入序列依然和普通Transformer保持一致，只不过把挖掉的一个词用"[MASK]"替换;

- Transformer 的 Encoder 部分按正常进行；

- 输出层在被挖掉的词位置，接一个分类层做词典大小上的分类问题，得到被 mask 掉的词概率大小。

- 正是因为加了 mask，因此 BERT 才把这种方法叫做 Masked-LM

![](../../images/week3/56.png)
    
    
这就直接把普通语言模型中的生成问题（正如 GPT 中把它当做一个生成问题一样，虽然其本质上也是一个序列生成问题），变为一个简单的分类问题，并且也直接解决了 Encoder 中多层 Self-attention 的双向机制带来的泄密问题（单层 Self-attention 是真双向，但不会带来泄密问题，只有多层累加的 Self-attention 才会带来泄密问题），使得语言模型中的真双向机制变为现实。


BERT 针对如何做“[MASK]”，做了一些更深入的研究，它做了如下处理：

- 选取语料中所有词的 15% 进行随机 mask；

- 选中的词在 80% 的概率下被真实 mask；

- 选中的词在 10% 的概率下不做 mask，而被随机替换成其他一个词；

- 选中的词在 10% 的概率下不做 mask，仍然保留原来真实的词。



#### Next Sentence Prediction

除了用上 Mask-LM 的方法使得双向 Transformer 下的语言模型成为现实，BERT 还利用和借鉴了 Skip-thoughts 方法中的句子预测问题，来学习句子级别的语义关系。具体做法则是将两个句子组合成一个序列，然后让模型预测这两个句子是否为先后近邻的两个句子，也就是会把"Next Sentence Prediction"问题建模__成为一个二分类问题__



在预训练阶段，因为有两个任务需要训练：Mask-LM 和 Next Sentence Prediction，因此 BERT 的预训练过程实质上是一个 Multi-task Learning

BERT的损失函数由两部分组成，第一部分是来自 Mask-LM 的单词级别分类任务，另一部分是句子级别的分类任务。通过这两个任务的联合学习，可以使得 __BERT 学习到的表征既有 token 级别信息，同时也包含了句子级别的语义信息__。具体损失函数如下:


![](../../images/week3/57.png)


- 其中 $\theta$ 是 BERT 中 Encoder 部分的参数;
- $\theta_1$ 是 Mask-LM 任务中在 Encoder 上所接的输出层中的参数;
- $\theta_2$ 则是 Next Sentence Prediction 任务中在 Encoder 接上的分类器参数


因此，在Masked-LM的损失函数中，如果被 mask 的词集合为 M，因为它是一个词典大小 $|V|$ 上的__多分类问题__，那么具体说来有:

![](../../images/week3/58.png)


在Next sentence prediction任务中，也是一个分类问题的损失函数:

![](../../images/week3/59.png)

两个任务联合学习的损失函数是:

![](../../images/week3/60.png)


BERT 还利用了一系列策略，使得模型更易于训练，比如对于学习率的 warm-up 策略（和上文提到的 ULMFiT 以及 Transformer 中用到的技巧类似），使用的激活函数不再是普通的 ReLu，而是 __GeLu__，也是用了 dropout 等常见的训练技巧


在输入层方面，思路和 GPT 基本类似，如果输入只有一个句子，则直接在句子前后添加句子的起始标记位和句子的结束符号。在 BERT 中，起始标记都用“[CLS]”来表示，结束标记符用"[SEP]"表示，对于两个句子的输入情况，除了起始标记和结束标记之外，两个句子间通过"[SEP]"来进行区分。

除了这些之外，BERT 还用两个表示当前是句子 A 或句子 B 的向量来进行表示。对于句子 A 来说，每一词都会添加一个同样的表示当前句子为句子 A 的向量，如果有句子 B 的话，句子 B 中的每个词也会添加一个表示当前句子为句子 B 的向量


![](../../images/week3/61.png)


除了输入层要尽量做到通用之外，根据不同任务设计不同的输出层也变得尤为重要，BERT 主要针对四类任务考虑和设计了一些非常易于移植的输出层，这四类任务分别是:
- 单个序列文本分类任务
- 两个序列文本分类任务
- 阅读理解任务和序列标注任务
- 句子或答案选择任务

对于单序列文本分类任务和序列对的文本分类任务使用框架基本一致，只要输入层按照上面的方法做好表示即可。

这两个分类任务都是__利用 Encoder 最后一层的第一个时刻“[CLS]”对应的输出作为分类器的输入__，这个时刻可以得到输入序列的全局表征，并且因为监督信号从这个位置反馈给模型，因而实际上在 finetune 阶段也可以使得这一表征尽量倾向于全局的表征。

在finetune 阶段，在 BERT Encoder 基础上，这些分类任务因为只需要接一个全连接层，因此增加的参数只有 $H × K$ ，其中 $H$ 是 Encoder 输出层中隐状态的维度，$K$ 是分类类别个数

对于 SQuAD 1.1 任务来说，需要在给定段落中找到正确答案所在区间，这段区间通过一个起始符与终止符来进行标记，因此只需__预测输入序列中哪个 token 所在位置是起始符或终止符即可__


![](../../images/week3/62.png)


Summary of BERT:

- 1. 双向功能是 Transformer Encoder 自带的，因此借鉴 Transformer；

- 2. Masked-LM 借鉴语言模型、CBOW 以及 Cloze问题；

- 3. Next Sentence Prediction 借鉴 Skip-gram、Skip-thoughts 和 Quick-thoughts 等工作；

- 4. 对输入层和输出层的改造，借鉴了 T-DMCA 以及 GPT 的做法

#### [BERT Code](https://github.com/brightmart/bert_language_understanding)

### Word2Vec using Gensim


```python
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
# Gensim only requires that the input must provide sentences sequentially, 
# when iterated over. No need to keep everything in RAM: provide one sentence, process it, forget it, load another sentence
model = gensim.models.Word2Vec(sentences, min_count=1)
```

    2018-11-23 23:42:34,891 : INFO : collecting all words and their counts
    2018-11-23 23:42:34,901 : INFO : PROGRESS: at sentence #0, processed 0 words, keeping 0 word types
    2018-11-23 23:42:34,903 : INFO : collected 3 word types from a corpus of 4 raw words and 2 sentences
    2018-11-23 23:42:34,911 : INFO : Loading a fresh vocabulary
    2018-11-23 23:42:34,915 : INFO : effective_min_count=1 retains 3 unique words (100% of original 3, drops 0)
    2018-11-23 23:42:34,921 : INFO : effective_min_count=1 leaves 4 word corpus (100% of original 4, drops 0)
    2018-11-23 23:42:34,925 : INFO : deleting the raw counts dictionary of 3 items
    2018-11-23 23:42:34,932 : INFO : sample=0.001 downsamples 3 most-common words
    2018-11-23 23:42:34,938 : INFO : downsampling leaves estimated 0 word corpus (5.7% of prior 4)
    2018-11-23 23:42:34,940 : INFO : estimated required memory for 3 words and 100 dimensions: 3900 bytes
    2018-11-23 23:42:34,948 : INFO : resetting layer weights
    2018-11-23 23:42:34,972 : INFO : training model with 3 workers on 3 vocabulary and 100 features, using sg=0 hs=0 sample=0.001 negative=5 window=5
    2018-11-23 23:42:35,024 : INFO : worker thread finished; awaiting finish of 2 more threads
    2018-11-23 23:42:35,031 : INFO : worker thread finished; awaiting finish of 1 more threads
    2018-11-23 23:42:35,034 : INFO : worker thread finished; awaiting finish of 0 more threads
    2018-11-23 23:42:35,035 : INFO : EPOCH - 1 : training on 4 raw words (0 effective words) took 0.0s, 0 effective words/s
    2018-11-23 23:42:35,046 : INFO : worker thread finished; awaiting finish of 2 more threads
    2018-11-23 23:42:35,049 : INFO : worker thread finished; awaiting finish of 1 more threads
    2018-11-23 23:42:35,052 : INFO : worker thread finished; awaiting finish of 0 more threads
    2018-11-23 23:42:35,054 : INFO : EPOCH - 2 : training on 4 raw words (0 effective words) took 0.0s, 0 effective words/s
    2018-11-23 23:42:35,061 : INFO : worker thread finished; awaiting finish of 2 more threads
    2018-11-23 23:42:35,065 : INFO : worker thread finished; awaiting finish of 1 more threads
    2018-11-23 23:42:35,068 : INFO : worker thread finished; awaiting finish of 0 more threads
    2018-11-23 23:42:35,070 : INFO : EPOCH - 3 : training on 4 raw words (1 effective words) took 0.0s, 99 effective words/s
    2018-11-23 23:42:35,080 : INFO : worker thread finished; awaiting finish of 2 more threads
    2018-11-23 23:42:35,083 : INFO : worker thread finished; awaiting finish of 1 more threads
    2018-11-23 23:42:35,085 : INFO : worker thread finished; awaiting finish of 0 more threads
    2018-11-23 23:42:35,088 : INFO : EPOCH - 4 : training on 4 raw words (0 effective words) took 0.0s, 0 effective words/s
    2018-11-23 23:42:35,113 : INFO : worker thread finished; awaiting finish of 2 more threads
    2018-11-23 23:42:35,116 : INFO : worker thread finished; awaiting finish of 1 more threads
    2018-11-23 23:42:35,120 : INFO : worker thread finished; awaiting finish of 0 more threads
    2018-11-23 23:42:35,125 : INFO : EPOCH - 5 : training on 4 raw words (0 effective words) took 0.0s, 0 effective words/s
    2018-11-23 23:42:35,127 : INFO : training on a 20 raw words (1 effective words) took 0.1s, 7 effective words/s
    2018-11-23 23:42:35,130 : WARNING : under 10 jobs per worker: consider setting a smaller `batch_words' for smoother alpha decay



```python
import os
word2vec = gensim.models.KeyedVectors.load_word2vec_format(fname='/home/karen/Downloads/data/glove.6B/glove.6B.100d.tmp.txt', binary=False)
word2vec
```

    2018-11-24 00:24:28,072 : INFO : loading projection weights from /home/karen/Downloads/data/glove.6B/glove.6B.100d.tmp.txt
    2018-11-24 00:26:45,137 : INFO : loaded (399999, 100) matrix from /home/karen/Downloads/data/glove.6B/glove.6B.100d.tmp.txt





    <gensim.models.keyedvectors.Word2VecKeyedVectors at 0x7f770ca3cd30>




```python
with open('/home/karen/Downloads/data/glove.6B/glove.6B.100d.txt', 'r') as fread:
    with open('/home/karen/Downloads/data/glove.6B/glove.6B.100d.tmp.txt', 'w') as fwrite:
        fwrite.write('399999 100\n')
        for idx, line in enumerate(fread):
            fwrite.write(line)
```


```python
!rm /home/karen/Downloads/data/glove.6B/glove.6B.100d.tmp.txt
```


```python
word2vec.init_sims(replace=True)
```

    2018-11-24 00:28:08,244 : INFO : precomputing L2-norms of word weight vectors



```python
len(word2vec.vocab)
```




    399999




```python
word2vec.vectors_norm[word2vec.vocab['student'].index]
```




    array([ 0.14084136,  0.10044176, -0.17062186, -0.03504317, -0.03259846,
            0.16226351,  0.00164017,  0.02982827, -0.04554551,  0.19863695,
           -0.08848356, -0.03040672,  0.06940352,  0.03717477, -0.05495993,
            0.05361199, -0.01390229, -0.01308273, -0.07149444,  0.08059222,
           -0.16011068, -0.03823438,  0.07592923, -0.01951167, -0.00767571,
           -0.14076705, -0.04336968, -0.19582428,  0.01398277,  0.07748415,
           -0.11785013,  0.21659192, -0.01383613,  0.07638032, -0.10138992,
           -0.1082482 , -0.03673607,  0.08005092, -0.05872075, -0.01275246,
           -0.147268  ,  0.00342861, -0.00873196,  0.02516351,  0.06088596,
           -0.05161836, -0.0100394 ,  0.03327952,  0.04757451,  0.04673779,
           -0.04099397, -0.11782714, -0.05391094,  0.07558075,  0.00579424,
           -0.30187368,  0.01918441, -0.0895591 ,  0.29233897,  0.04310788,
           -0.00594   ,  0.01523573, -0.04347936, -0.01092671,  0.11162515,
            0.01602999,  0.13726982,  0.06173506,  0.18407837,  0.16519998,
            0.03331666, -0.04278062, -0.02724027, -0.04862705, -0.09478815,
            0.13880351,  0.11250433, -0.09805897, -0.09296789, -0.17817003,
            0.055572  ,  0.04316802, -0.00249795, -0.22557826, -0.25593367,
           -0.04909405, -0.02562875, -0.08425928,  0.07797062, -0.1113262 ,
            0.04604082,  0.04001042,  0.06554011,  0.08327927, -0.09876832,
            0.07073379,  0.02987957, -0.05065782,  0.0969799 , -0.05943188],
          dtype=float32)



#### Sentence2Vec by word2vec


```python
def create_word2vec_matrix(text, word2vec):
    word2vec_matrix=[]
    count=0
    for line in text:
        word_lst=line.split()
        current_word2vec=[]
        for word in word_lst:
            if word in word2vec.vocab:
                # word2vec = token2idx
                vec = word2vec.vectors_norm[word2vec.vocab[word].index]
                if vec is not None:
                    current_word2vec.append(vec)
            else:
                print(word)
                count+=1
                continue
        # add up all the vector of each word to get the vector of a sentence 
        if np.array(current_word2vec).shape[0]!=0:
            sentence_word2vec = list(np.array(current_word2vec).mean(axis=0))
            word2vec_matrix.append(sentence_word2vec)
        current_word2vec=[]
    return word2vec_matrix, count
```


```python
text = ['fantastic beasts and where to find them', 
        'fantastic beasts the crimes of grindelwald']
word2vec_matrix, count = create_word2vec_matrix(text, word2vec)
```


```python
np.array(word2vec_matrix)
```




    array([[-0.01363225,  0.0632612 ,  0.0532789 , -0.06067752, -0.03203793,
             0.0573948 , -0.05758385,  0.02891669, -0.01005686, -0.04475319,
             0.03142466, -0.00525038,  0.03889203, -0.01838762,  0.04951621,
            -0.0357757 ,  0.04723538,  0.06406571, -0.110898  ,  0.08045464,
             0.06799715, -0.03494107,  0.05513481, -0.04234928,  0.04503259,
            -0.0038799 , -0.02963971, -0.0674924 ,  0.0407795 , -0.05591179,
            -0.03720056,  0.01724851, -0.06198395, -0.00394624,  0.01963765,
             0.04357598, -0.03199228,  0.04052764,  0.01646447, -0.06384147,
            -0.06349035, -0.01205016, -0.03160828, -0.10188204, -0.02248716,
             0.05475311, -0.00057904, -0.00529261,  0.00310575, -0.06026373,
            -0.03455113,  0.0093182 ,  0.04103724,  0.22216587, -0.00798338,
            -0.37076157,  0.0060763 , -0.02901358,  0.19122097,  0.05265542,
            -0.0445339 ,  0.15990758, -0.07531472,  0.02351545,  0.13658723,
            -0.00910989,  0.08402754,  0.06993966,  0.04774452, -0.07440751,
             0.00432026, -0.1105385 ,  0.01313347, -0.07487839,  0.02296722,
             0.0301899 , -0.0280746 , -0.01931024, -0.08416656,  0.03807382,
             0.1088913 ,  0.05673665, -0.07269619,  0.04082172, -0.2123229 ,
            -0.03421097,  0.00801395, -0.01705644, -0.07357238, -0.03133134,
            -0.01436853, -0.05166963,  0.02597543, -0.01765687, -0.12825742,
            -0.05672801, -0.09524342, -0.05342931,  0.0833168 ,  0.04327292],
           [ 0.00253851, -0.00877398,  0.0734567 , -0.00424336,  0.00065437,
             0.05516574, -0.07829381,  0.01089091, -0.05188168, -0.02742841,
             0.02640177,  0.01450567, -0.00968969, -0.03183353,  0.04959001,
             0.02107229,  0.01465111,  0.03763356, -0.07321229,  0.02437557,
             0.0532227 , -0.02813247, -0.01158483, -0.04175212,  0.03454379,
            -0.02026052,  0.03413947, -0.00275082, -0.01641666, -0.04765847,
             0.02100672, -0.03880749, -0.03762351, -0.04919778,  0.00471069,
            -0.00720015, -0.01010328,  0.0680422 ,  0.0268242 , -0.07889139,
            -0.05303546,  0.02562189,  0.00818047, -0.0245726 ,  0.00355374,
             0.08366998,  0.06239701, -0.02762469, -0.01287852, -0.00160172,
            -0.00158122,  0.01933938,  0.03472779,  0.13059537, -0.0444909 ,
            -0.20608287,  0.01566952,  0.01300212,  0.10512716,  0.02248251,
            -0.01267021,  0.11147016, -0.04865355, -0.00529323,  0.08803131,
             0.01013602,  0.01064846,  0.01560272,  0.03319718, -0.03983454,
             0.04644413, -0.10616875, -0.00987621,  0.05064338,  0.01425096,
             0.02025635, -0.02772122, -0.01244517, -0.07074878,  0.04453899,
             0.09321102,  0.10945683, -0.02349626,  0.04435547, -0.16061288,
            -0.00520629, -0.0616868 , -0.02149554,  0.00955985, -0.01845567,
            -0.02264366, -0.02768158,  0.03380519,  0.06865028, -0.03667846,
            -0.05988262, -0.10793936, -0.03674577,  0.06783447,  0.01057718]],
          dtype=float32)




```python
count # so all words are in vocab
```




    0




```python
sentence1 = word2vec_matrix[0]
sentence2 = word2vec_matrix[1]
from scipy import spatial

# calculate the cosine similarity
1 - spatial.distance.cosine(sentence1, sentence2)
```




    0.8124398589134216



#### Most similar words


```python
word2vec.most_similar(positive=['kindergarten', 'college'], negative=['scientist'], topn=10)
```

    /usr/local/lib/python3.6/dist-packages/gensim/matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int64 == np.dtype(int).type`.
      if np.issubdtype(vec.dtype, np.int):





    [('school', 0.7116979360580444),
     ('elementary', 0.7066525220870972),
     ('grades', 0.7055771350860596),
     ('schools', 0.6759970188140869),
     ('preschool', 0.6746401786804199),
     ('pupils', 0.6707763075828552),
     ('classes', 0.646867036819458),
     ('schooling', 0.6365389227867126),
     ('vocational', 0.6250067949295044),
     ('enrollment', 0.6218675374984741)]



#### Find the different word in a sentence


```python
word2vec.doesnt_match("breakfast cereal dinner lunch".split())
```

    /usr/local/lib/python3.6/dist-packages/gensim/matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int64 == np.dtype(int).type`.
      if np.issubdtype(vec.dtype, np.int):





    'cereal'



#### calculate the similarity between two words


```python
word2vec.similarity('apocalypse', 'disaster')
```

    /usr/local/lib/python3.6/dist-packages/gensim/matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int64 == np.dtype(int).type`.
      if np.issubdtype(vec.dtype, np.int):





    0.1993172



### [Word2Vec using Fasttext](https://pypi.org/project/fasttext/)

#### FastText?

word2vec 和 GloVe 都不需要人工标记的监督数据，只需要语言内部存在的监督信号即可以完成训练。而与此相对应的，__fastText 是利用带有监督标记的文本分类数据完成训练__

本质上没有什么特殊的，__模型框架就是 CBOW__，只不过与普通的 CBOW 有两点不一样，分别是__输入数据和预测目标的不同__:

- 在输入数据上，CBOW 输入的是一段区间中除去目标词之外的所有其他词的向量加和或平均，__而 fastText 为了利用更多的语序信息，将 bag-of-words 变成了 bag-of-features__，也就是输入 x 不再仅仅是一个词，还可以加上 bigram 或者是 trigram 的信息等等。


![](../../images/week3/36.png)

- 在预测目标上，CBOW 预测目标是语境中的一个词，而 __fastText 预测目标是当前这段输入文本的类别__，正因为需要这个文本类别，因此才说 fastText 是一个监督模型。

而相同点在于，fastText 的网络结构和 CBOW 基本一致，同时在输出层的分类上也使用了 Hierachical Softmax 技巧来加速训练。

这里的$x_{n,i}$便是语料当中第 n 篇文档的第 i 个词以及加上 N-gram 的特征信息。从这个损失函数便可以知道 __fastText 同样只有两个全连接层，分别是 A 和 B__，其中 A 便是最终可以获取的词向量信息


```python
import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

load_vectors('/home/karen/Downloads/data/wiki-news-300d-1M.vec')
```


```python
import fasttext

# Skipgram model
model = fasttext.skipgram('toy_data/second.txt', 'model')
model['fantastic']
```




    [-0.000991184264421463,
     -0.001029246486723423,
     0.0006811252678744495,
     0.00011803481174865738,
     -0.0006219972274266183,
     0.001100853318348527,
     0.00014544253644999117,
     -0.0008069276809692383,
     0.0008861601236276329,
     0.0005640610470436513,
     0.00018239919154439121,
     0.002081118058413267,
     0.0011119148693978786,
     -8.017678737815004e-06,
     0.0019191585015505552,
     0.002273016609251499,
     0.0004999999655410647,
     0.0008727581007406116,
     0.0017401062650606036,
     0.00136960344389081,
     -0.00010665664740372449,
     0.0006562909111380577,
     -0.0005103441653773189,
     0.0006105859065428376,
     -0.001563229481689632,
     0.0024708015844225883,
     -7.241084676934406e-05,
     0.00035496175405569375,
     -0.0008005818235687912,
     0.001430544420145452,
     0.0005475004436448216,
     -0.000570164353121072,
     0.00010769572691060603,
     -2.468213642714545e-05,
     0.0015175550943240523,
     0.0003394550003577024,
     -0.0006585742812603712,
     -0.0010379229206591845,
     0.0002934975200332701,
     -0.0004705099854618311,
     -0.0005588103667832911,
     -0.0001224353618454188,
     0.00018472163355909288,
     -0.0008804462268017232,
     -0.000540959823410958,
     0.0007997071370482445,
     -0.000813662598375231,
     -0.0012080087326467037,
     0.00136748724617064,
     0.0007802385371178389,
     0.0003648688143584877,
     -3.339976683491841e-05,
     -0.00031441979808732867,
     -0.0003039219882339239,
     -0.0005072257481515408,
     0.0010179569944739342,
     -0.0009995679138228297,
     -0.0003505126223899424,
     0.0003821019490715116,
     0.0011511098127812147,
     -0.0005719128530472517,
     -0.0006986728403717279,
     0.0005401912494562566,
     -0.0013313741656020284,
     0.00041330300155095756,
     -0.0006128869135864079,
     0.0012367976596578956,
     -0.001043042866513133,
     0.000432994042057544,
     -0.0024143403861671686,
     -0.00034675822826102376,
     -0.00010233062494080514,
     0.001215107273310423,
     0.0012985324719920754,
     -6.912585376994684e-05,
     -2.201867027906701e-05,
     -0.0011087505845353007,
     -0.0006296449573710561,
     0.0004131598980166018,
     0.0010789132211357355,
     0.001263558049686253,
     0.00032109773019328713,
     0.0006544655188918114,
     -0.0001419403706677258,
     -0.00021072222443763167,
     0.00044179518590681255,
     0.0007763911853544414,
     0.0009751742472872138,
     -0.0004021052736788988,
     -0.0007130807498469949,
     -0.001583990640938282,
     -0.0005740714841522276,
     -0.0015615387819707394,
     -0.0025593223981559277,
     0.00018674305465538055,
     0.0009086879435926676,
     0.0008893464109860361,
     0.00039883964927867055,
     -0.0007123534451238811,
     0.0025303619913756847]




```python
# CBOW model
model = fasttext.cbow('toy_data/second.txt', 'model')
model['fantastic']
```




    [-0.000991184264421463,
     -0.001029246486723423,
     0.0006811252678744495,
     0.00011803481174865738,
     -0.0006219972274266183,
     0.001100853318348527,
     0.00014544253644999117,
     -0.0008069276809692383,
     0.0008861601236276329,
     0.0005640610470436513,
     0.00018239919154439121,
     0.002081118058413267,
     0.0011119148693978786,
     -8.017678737815004e-06,
     0.0019191585015505552,
     0.002273016609251499,
     0.0004999999655410647,
     0.0008727581007406116,
     0.0017401062650606036,
     0.00136960344389081,
     -0.00010665664740372449,
     0.0006562909111380577,
     -0.0005103441653773189,
     0.0006105859065428376,
     -0.001563229481689632,
     0.0024708015844225883,
     -7.241084676934406e-05,
     0.00035496175405569375,
     -0.0008005818235687912,
     0.001430544420145452,
     0.0005475004436448216,
     -0.000570164353121072,
     0.00010769572691060603,
     -2.468213642714545e-05,
     0.0015175550943240523,
     0.0003394550003577024,
     -0.0006585742812603712,
     -0.0010379229206591845,
     0.0002934975200332701,
     -0.0004705099854618311,
     -0.0005588103667832911,
     -0.0001224353618454188,
     0.00018472163355909288,
     -0.0008804462268017232,
     -0.000540959823410958,
     0.0007997071370482445,
     -0.000813662598375231,
     -0.0012080087326467037,
     0.00136748724617064,
     0.0007802385371178389,
     0.0003648688143584877,
     -3.339976683491841e-05,
     -0.00031441979808732867,
     -0.0003039219882339239,
     -0.0005072257481515408,
     0.0010179569944739342,
     -0.0009995679138228297,
     -0.0003505126223899424,
     0.0003821019490715116,
     0.0011511098127812147,
     -0.0005719128530472517,
     -0.0006986728403717279,
     0.0005401912494562566,
     -0.0013313741656020284,
     0.00041330300155095756,
     -0.0006128869135864079,
     0.0012367976596578956,
     -0.001043042866513133,
     0.000432994042057544,
     -0.0024143403861671686,
     -0.00034675822826102376,
     -0.00010233062494080514,
     0.001215107273310423,
     0.0012985324719920754,
     -6.912585376994684e-05,
     -2.201867027906701e-05,
     -0.0011087505845353007,
     -0.0006296449573710561,
     0.0004131598980166018,
     0.0010789132211357355,
     0.001263558049686253,
     0.00032109773019328713,
     0.0006544655188918114,
     -0.0001419403706677258,
     -0.00021072222443763167,
     0.00044179518590681255,
     0.0007763911853544414,
     0.0009751742472872138,
     -0.0004021052736788988,
     -0.0007130807498469949,
     -0.001583990640938282,
     -0.0005740714841522276,
     -0.0015615387819707394,
     -0.0025593223981559277,
     0.00018674305465538055,
     0.0009086879435926676,
     0.0008893464109860361,
     0.00039883964927867055,
     -0.0007123534451238811,
     0.0025303619913756847]




```python
model = fasttext.load_model('model.bin')
# model['fantastic']
```


```python

```
