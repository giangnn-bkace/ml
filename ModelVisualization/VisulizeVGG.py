from keras import applications
from keras import backend as K
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_width = 128
img_height = 128

# Load model
model = applications.VGG16(include_top=False, weights='imagenet')

layer_dict = dict([(layer.name, layer) for layer in model.layers])
input_img = model.input

# build a loss function that maximizes the activation of the nth filter of the layer considered
layer_name = 'block1_conv2' # name of the layer
filter_index = 10 # can be any integer from 0 to #number of filters in the layer
layer_output = layer_dict[layer_name].output
loss = K.mean(layer_output[:, :, :, filter_index])
grads = K.gradients(loss, input_img)[0] # compute the gradient of the input picture wrt this loss
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) # normalization the gradient
iterate = K.function([input_img], [loss, grads]) # returns the loss and grads given the input picture

# gradient ascent
step = 1.
input_img_data = (np.random.random((1, img_width, img_height, 3))-0.5) * 20 + 128.
for i in range(20): # run gradient ascent for 20 steps
	loss_value, grads_value = iterate([input_img_data])
	input_img_data += grads_value * step

# visualize the filter
def deprocess_image(x):
	# normalize tensor: center on 0., ensure std is 0.1
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1
	# clip to [0, 1]
	x += 0.5
	x = np.clip(x, 0, 1)
	# convert to RGB array
	x *= 255
	#x = x.transpose((1, 2, 0))
	x = np.clip(x, 0, 255).astype('uint8')
	return x

img = input_img_data[0]
img = deprocess_image(img)
imsave('%s_filter_%d.png' % (layer_name, filter_index), img)