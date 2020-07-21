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
