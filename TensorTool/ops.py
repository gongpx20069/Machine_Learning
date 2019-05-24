import tensorflow as tf

def Ck(input,k_size=3,k, slope=0.2, stride=2, reuse=False, norm='instance', is_training=True, name=None,padding="SAME"):
	""" A 3x3 Convolution-BatchNorm-LeakyReLU layer with k filters and stride 2
	Args:
		input: 4D tensor
    		k_size: The size of filter
		k: integer, number of filters (output depth)
		slope: LeakyReLU's slope
		stride: integer
		norm: 'instance' or 'batch' or None
		is_training: boolean or BoolTensor
		reuse: boolean
		name: string, e.g. 'C64'
    		padding:'SAME' or 'VALID'
	Returns:
		4D tensor
	"""
	with tf.variable_scope(name, reuse=reuse):
		weights = _weights("weights",
		shape=[k_size, k_size, input.get_shape()[3], k])

		conv = tf.nn.conv2d(input, weights,
			strides=[1, stride, stride, 1], padding=padding)

		normalized = _norm(conv, is_training, norm)
		output = _leaky_relu(normalized, slope)
		return output
    
    
def Dense(input,slope=0.2,norm='instance',is_training=True,reuse = False,name = None,units=1024,activation='leakyrelu'):
	"""
		A dense layer with leaky relu
	Args:
		input:4D tensor
    		slope: LeakyReLU's slope
		reuse:boolean
		use_sigmoid:boolean
    		norm: 'instance' or 'batch' or None
    		is_training: boolean or BoolTensor
		name:string,e.g. 'dense'
    		activation: 'leakyrelu' or 'sigmoid' or None
		units:output layer point number
	Returns:
		1D tensor

	"""
	with tf.variable_scope(name, reuse=reuse):
		den = tf.layers.dense(inputs=input,units=units)
		normalized = _norm(den,is_training,norm)
		if activation=='leakyrelu':
			output = _leaky_relu(normalized,slope)
		elif activation=='sigmoid':
			output=tf.sigmoid(normalized)
		else:
			output = normalized
    return output



### Helpers
def _weights(name, shape, mean=0.0, stddev=0.02):
	""" Helper to create an initialized Variable
	Args:
		name: name of the variable
		shape: list of ints
		mean: mean of a Gaussian
		stddev: standard deviation of a Gaussian
	Returns:
		A trainable variable
	"""
	var = tf.get_variable(
		name, shape,
		initializer=tf.random_normal_initializer(
		mean=mean, stddev=stddev, dtype=tf.float32))
	return var

def _biases(name, shape, constant=0.0):
	""" Helper to create an initialized Bias with constant
	"""
	return tf.get_variable(name, shape,
			initializer=tf.constant_initializer(constant))

def _leaky_relu(input, slope):
	return tf.maximum(slope*input, input)

def _norm(input, is_training, norm='instance'):
	""" Use Instance Normalization or Batch Normalization or None
	"""
	if norm == 'instance':
		return _instance_norm(input)
	elif norm == 'batch':
		return _batch_norm(input, is_training)
	else:
		return input

def _batch_norm(input, is_training):
	""" Batch Normalization
	"""
	with tf.variable_scope("batch_norm"):
		return tf.contrib.layers.batch_norm(input,
										decay=0.9,
										scale=True,
										updates_collections=None,
										is_training=is_training)

def _instance_norm(input):
	""" Instance Normalization
	"""
	with tf.variable_scope("instance_norm"):
		depth = input.get_shape()[3]
		scale = _weights("scale", [depth], mean=1.0)
		offset = _biases("offset", [depth])
		mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
		epsilon = 1e-5
		inv = tf.rsqrt(variance + epsilon)
		normalized = (input-mean)*inv
		return scale*normalized + offset

def safe_log(x, eps=1e-12):
	return tf.log(x + eps)
