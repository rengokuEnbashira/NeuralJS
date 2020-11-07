# NeuralJS
This is a simple fully connected neural network library implemented in Javascript.
## How to use it?
First we need to create a neural_network object
```js
let model = new neural_network();
```
Then we add blocks (a block connects 2 consecutive layers)
```js
// inputs for the block constructor
// number of inputs, number of outputs, activation function
model.add(new block(2,5,"tanh");
model.add(new block(5,2,"linear");
```
We can choose from the following activation function:

* sigmoid
* tanh 
* relu
* leaky_relu
* linear

After we add all the blocks that we need, we proceed to train the model with the train data.
```js
//inputs for training:
//inputa_data,target_data,loss function, maximum number of iteration,learning rate
model.train(input_data,target_data,"cross_entropy",1000,1);
```
We can choose from the following loss functions:
* mean_square_error
* cross_entropy

