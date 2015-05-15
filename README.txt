I tried to make the code as simple and readable as possible, and documented 
the majority of it, in addition Python is naturally quite simple to read, so 
hopefully you should be able to get what's going on in the code even without 
knowing the language.

##############################################################################

				CODE STRUCTURE

##############################################################################

in NDA you have all the classes to make an NDA:
- A class implementing the generalized shift
- A class implementing a Godel encoder
- A class implementing the NDA from a generalized shift and a Godel encoder

In simpleNNlib.py I implemented some stuff to build simple neural networks with 
saturated-linear activation function and Heaviside activation function.

In NeuralTM.py I implement the Neural Network we are presenting in the paper,
ugin simpleNNlib.py to make the neurons and connections, and NDA.py to get the 
weights for the linear transformation layer.

In examplenet.py I actually use these libraries to reproduce Peter's example
parser.


