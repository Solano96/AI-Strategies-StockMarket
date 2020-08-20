# Artificial Intelligence Trading

App to test strategies based on artificial intelligence for investing in the stock market.

The program has two simple investment strategies to compare results. One of these strategies is simply to buy and hold. The other is a classic strategy based on the crossing of Moving Averages and the use of the Relative Strength Index or RSI.

At this moment the app has the following strategies based on artificial intelligence:
* [Deep Neural Network](https://solano96.github.io/AI-Strategies-StockMarket/docs/neural_network): strategy that tries to predict the market trend with the use of neural networks that take different technical indicators as inputs.
* Strategy that combines in a weighted way buy-sell signals coming from moving average crosses. The weights are obtained through the PSO (Particle swarm optimization) algorithm.

## Getting Started 🚀

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. The local installation has been successfully tested in Ubuntu 18.04.

### Prerequisites 📋

Have installed Python3, you can check with the following command in your terminal:

```
python3 -V
```

In case you did not have Python3 installed, you can use the following commands:

```
sudo apt-get update
sudo apt-get install python3
```

To use the program with interface is necessary to intall tkinter with the following command:
```
sudo apt-get install python3-tk
```

### Installing 🔧

First clone the repository:
```
git clone https://github.com/Solano96/AI-Strategies-StockMarket.git
```

Now we need to install some dependencies. To do this, execute the following command:

```bash
sh scripts/install_local.sh
```

## Usage 📦

First we need to activate the virtual environment with:

```bash
. venv/bin/activate
```

You can use the command line program, that can be execute as follow:

```
python3 main.py --strategy <strategy-name> --quote <quote-name> --from-date <from> --to-date <to>
```

You can also use the short options:

```
python3 main.py -s <strategy-name> -q <quote-name> -f <from> -t <to>
```

Example:

```
python3 main.py --strategy neural-network --quote AAPL --from-date 2011-12-22 --to-date 2013-12-22
```

PD: You can use as data name any market abbreviation recognized by yahoo finance.

After the execution you can find the results in 'resultados' folder with the different strategies and you can see the graph of simulation in img folder.

The other alternative is to use the version with interface, this version can be execute with the following command:

```
python3 interface.py
```

<img src="docs/interfaz.png"
     style="float: left; margin-right: 10px;" />

## Author ✒️

* **Francisco Solano López Rodríguez**
