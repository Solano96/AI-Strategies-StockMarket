# Neural Networks Trading

Program to predict the market trend using neural networks.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Have installed Python3, you can check with the following command in your terminal:

```
python3 -V
```

In case you did not have Python3 installed, you can use the following commands:

```
sudo apt-get update
sudo apt-get install python3
```

Have pip3 installed, pip3 is a package manager for Python packages

```
sudo apt-get update
sudo apt-get -y install python3-pip
```

To use the program with interface is necessary to intall tkinter with the following command:
```
sudo apt-get install python3-tk
```

### Installing

First clone the repository:
```
git clone https://github.com/Solano96/TFG.git
```

Once you have downloaded the repository, move to the folder with *cd*.
Now, you have to activate the virtual environment:

```
source env/bin/activate
```

Then run the following command to install needed libs:
```
pip3 install -r requirements.txt
```

## Usage

You can use the command line program, that can be execute as follow:

```
python3 main.py <data-name> <gain> <loss> <simulation days> <training-epochs>
```

Example:

```
python3 main.py SAN 0.07 0.03 20 300
```

PD: You can use as data name any market abbreviation recognized by yahoo finance.

After the execution you can find the results in 'resultados' folder with the different strategies and you can see the graph of simulation in img folder.

The other alternative is to use the version with interface, this version can be execute with the following command:

```
python3 interface.py
```

<img src="img/interfaz.png"
     style="float: left; margin-right: 10px;" />

## Author

* **Francisco Solano López Rodríguez**
