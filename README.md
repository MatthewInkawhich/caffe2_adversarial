This repo serves as a code sharing mechanism for research between Nathan and Matthew Inkawhich.


##############
Contents:
##############
tools: Scripts that take do not require modifying code itself; all variables are given as command line args.
	- convert_imageset.py: Extension of lmdb_create_example.py from Caffe2 docs. This file creates an lmdb of raw jpg images using a labels file.

train: Scripts to facilitate creation, training, and saving custom nets. Requires modifying code directly.
	- train_mnist.py: Extension of MNIST tutorial from Caffe2 docs.
