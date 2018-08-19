
## Intro to CNN 

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/2.png)

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/3.png)

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/4.png)

## Transfer Learning and Finetuning

Deep Networks learn complex features extractor, but need lots of data to train it from scratch

> What if we can reuse an existing features extractor for a new task?

### Transfer Learning 
- need much less data to train (for training only final MLP)
- It works if a domain of a new task is similar to ImageNet (__How to evaluate this similarity? Is there a quantifiable method?__)
    - For Example:
        - Will not work for human emotion classification since there's no human faces in ImageNet dataset
        - In this case, we should partially reuse the ImageNet extractor by __FineTuning__

### FineTuning

- Instead of starting with a random initialization, initialize deeper layers with values from ImageNet
- Propagate all gradients with smaller learning rate
- __Keras has the weights of pre-trained VGG, Inception, ResNet architectures__
- We can fine-tune a bunch of different architectures and make an ensemble out of them

____

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/5.png)


____
### Other tasks (explain in later notes)
![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/6.png)


#### Unpooling Methods

- Nearest neighbor unpooling

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/7.png)

- Max unpooling

![](https://raw.githubusercontent.com/karenyyy/Advanced_ML_HSE/master/Introduction-To-Deep-Learning/images/8.png)

