
## Language Models

### Count N-gram language models

![](../../images/39.png)

![](../../images/40.png)

----

#### LM Applications

![](../../images/41.png)


#### chain rule:

$$p(w) = p(w_1) p(w_2 | w_1)... p(w_k| w_1...w_{k-1})$$


#### Markov assumption

- we don't need to care about all the history, just extract part of it


$$p(w_i | w_1 ... w_{i-1}) = p(w_i| w_{i-n+1}...w_{i-1})$$


#### Bigram language model

![](../../images/42.png)

![](../../images/43.png)

![](../../images/44.png)





#### Ngram model

![](../../images/50.png)

![](../../images/51.png)



__Example:__

![](../../images/52.png)

![](../../images/53.png)






### Perplexity

> How to train n-gram models?

Log-likelihood maximization:

$$\log p(W_{train}) = \sum_{i=1}^{N+1} \log p(w_i | w^{i-1}_{i-n+1}) \rightarrow \max$$

where

$N$ is the length of the train corpus

$$ p(w_i | w^{i-1}_{i-n+1}) = 
\frac{c(w^i_{i-n+1})}{\sum_{w_i} c(w^i_{i-n+1})} = \frac{c(w^i_{i-n+1})}{\sum_{w_i} c(w^{i-1}_{i-n+1})}$$


![](../../images/54.png)



- Evaluate the model on the test set

    - likelihood
    
    $$L = p(w_{test}) = \prod_{i=1}^{N+1} p(w_i| w^{i-1}_{i-n+1})$$
    
    - perplexity:
    
    $$P = p(w_{test})^{-\frac{1}{N}} = \frac{1}{\sqrt[N]{p(w_{test})}}$$
    



    


### Additional:

#### perplexity

所谓__语言模型（Language Model，LM__，即给定一句话的前k个词，我们希望语言模型可以预测第k+1个词是什么，即给出一个第k+1个词可能出现的概率的分布 $p(x_{k+1}|x_1, x_2...x_k)$

----

`perplexity`指标:

衡量一个语言模型的好坏，根据与语言模型自身的一些特性，的一种简单易行，而又行之有效的评测指标

简单来说，perplexity就是对于语言模型所估计的一句话出现的概率，用句子长度normalize一下，具体形式如下

![](../../images/45.png)


__Perplexity越小越好，相应的，就是我们见过的句子出现的概率越大越好__


Perplexity其实表示的是average branch factor，大概可以翻译为平均分支系数。即平均来说，我们预测下一个词时有多少种选择

举个例子，对于一个长度为N的，由0-9这10个数字随机组成的序列。由于这10个数字随机出现，所以每个数字出现的概率是110。也就是，在每个点，我们都有10个等概率的候选答案供我们选择，于是我们的perplexity就是10（有10个合理的答案）。具体计算过程如下:

![](../../images/46.png)






###  Smoothing in N-Gram

##### Add-one (Laplace) Smoothing 


Add-one是最简单、最直观的一种平滑算法。

__既然希望没有出现过的N-Gram的概率不再是0，那就不妨规定任何一个N-Gram在训练语料至少出现一次__，则:

$$count_{new}(n-gram)=count_{old}(n-gram)+1$$

for unigram model:

$$P_{add_one}(w_i) = \frac{C(w_i)+1}{M+|V|}$$

$$M: occurrence \: frequency \: of \: w_i \: in \: tokens$$

$$V: len(set(tokens))$$


for bigram model:

$$P_{add_one}(w_i|w_{i-1}) = \frac{C(w_{i-1}w_i)+1}{C(w_{i-1})+|V|}$$

for ngram model:

$$P_{add_one}(w_i|w_{i-n+1}, ..., w_{i-1}) = \frac{C(w_{i-n+1}, ..., w_{i-1}, w_i)+1}{C(w_{i-n+1}, ..., w_{i-1})+|V|}$$

__Example:__

sentence: `<s>the rat ate the cheese</s>`

$$P(ate|rat) = \frac{C(ate, rat)+1}{C(rat) + V} = \frac{1+1}{1+5} = \frac{2}{6}$$

$$P(ate|cheese) = \frac{C(ate, cheese)+1}{C(cheese) + V} = \frac{0+1}{1+5} = \frac{1}{6}$$




##### Add-k Smoothing（Lidstone’s law）

$$P_{add_k}(w_i|w_{i-n+1}, ..., w_{i-1}) = \frac{C(w_{i-n+1}, ..., w_{i-1}, w_i)+k}{C(w_{i-n+1}, ..., w_{i-1})+k|V|}$$

__Note:__

通常，add-k算法的效果会比Add-one好，但是显然它不能完全解决问题。至少在实践中，k 必须人为给定



##### Add-one，以及Add-k算法。策略本质上说其实是将一些频繁出现的 N-Gram 的概率匀出了一部分，分给那些没有出现的 N-Gram 上。因为所有可能性的概率之和等于1，所以我们只能在各种可能的情况之间相互腾挪这些概率

##### 回退（Backoff）

![](../../images/55.png)

在高阶模型可靠时，尽可能的使用高阶模型。但是有时候高级模型的计数结果可能为0，这时我们就转而使用低阶模型来避免稀疏数据的问题

__Example:__

统计语料库中 “like chinese food” 出现的次数，结果发现它没出现过，则计数为0。在回退策略中，将会__试着用低阶gram来进行替代__，也就是用 “chinese food” 出现的次数来替代



##### 插值（Interpolation）

在使用插值算法时，我们把不同阶别的n-Gram模型线形加权组合后再来使用。简单线性插值（__Simple Linear Interpolation__）可以用下面的公式来定义:

![](../../images/48.png)

where 

$$0 \le \lambda \le 1, \sum_i \lambda_i = 1$$

__Note:__

$\lambda_i$ 可以根据试验凭经验设定，也可以通过应用某些算法确定，例如$EM$算法

在简单单线形插值法中，权值 $\lambda_i$ 是常量。问题在于不管高阶模型的估计是否可靠（毕竟有些时候高阶的Gram计数可能并无为 0），__低阶模型均以同样的权重被加入模型，这并不合理__。一个可以想到的解决办法是让 $\lambda_i$ 成为历史的函数。如果__用递归的形式重写__插值法的公式，则有:


- Jelinek-Mercer Smoothing

![](../../images/49.png)




##### Absolute discounting

- idea: compare the counts for bigrams in train and test sets



### Sequence tagging with probabilistic models / DL

![](../../images/week2/1.png)

![](../../images/week2/2.png)

![](../../images/week2/3.png)

![](../../images/week2/4.png)

![](../../images/week2/5.png)

![](../../images/week2/6.png)

![](../../images/week2/7.png)


![](../../images/week2/8.png)

![](../../images/week2/9.png)


