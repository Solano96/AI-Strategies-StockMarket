# Deep Neural Network

## Definition of Multilayer Neural Network

The neurons of the multilayer perception are organised in layers of different levels. Three types of layers can be distinguished: the input layer, the hidden layers and the output layer.

The neurons in the input layer are only responsible for receiving signals from the outside and propagating them to all the neurons in the next layer. The last layer is the output layer, which provides the outside with the neural network response for each of the patterns that the network has received as input. The neurons in the hidden layers perform non-linear processing of the input data.

In the multilayer perception the connections are directed forward, so that neurons in one layer connect to neurons in the next layer. As in the simple perception the connections between neurons have an associated weight and the neurons have an associated threshold. The threshold is usually treated as if it were one more connection with constant input and equal to 1.

The network is said to have total connectivity because the neurons in one layer are connected to all the neurons in the next layer.

## Architecture of our model

The implemented neural network consists of 6 layers. The first layer corresponds to the input layer and has a total of 36 nodes. The last layer is the output layer which has a single node, whose output will predict the market trend depending on whether the output value is closer to -1 or +1. The inner layers correspond to the hidden layers of the network and have 128, 64, 16 and 8 nodes respectively. 
The input values of the neural network correspond to the following attributes.

* Open, close, low, high and volume.
* Momentum of 5, 10 and 15 days.
* Simple moving average of 7, 14 and 21 days.
* Exponential moving average of 7, 14 and 21 days.
* Rate of change of 13 and 21 days.
* Stochastic oscillator ( %K, %D) of 7, 14 and 21 days.
* Fast stochastic oscillator ( %K, %D) of 7, 14 and 21 days.
* MACD and histogram, with difference of moving averages of 12 and 26 periods.
* Relative Strength Index of 9, 14 and 21 days.
* Standard deviation of 7, 14 and 21 days.
