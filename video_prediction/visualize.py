import imageio
import numpy as np

# include the background
num_masks = 10 + 1
num_time_steps = 9

masks = np.load('masks.npy')

# switch to (timestep, layer, batch_id, row, col) index
masks = masks.reshape((9, num_masks, 32, 64, 64))

kernels = np.load('kernels.npy')

# switch to (timestep, layer, batch_id, row, col) index
# all color channel share the same mask
kernels = kernels[:, :, 0, :, :, 0, :].transpose([0, 4, 1, 2, 3])

tf_layers = np.load('tf_layers.npy')

tf_layers = tf_layers.reshape((9, num_masks, 32, 64, 64, 3))

gen_images = np.load('gen_images.npy')
gt_images = np.load('gt_images.npy')


for i in xrange(32):
    imageio.mimsave('img.%d.gif' % i, np.hstack([gen_images[:, i], gt_images[1:, i]]))

flat_masks = (masks/masks.max(axis=(3, 4), keepdims=True)).transpose((0, 2, 3, 1, 4))
for i in xrange(32):
    imageio.mimsave('masks.%d.gif' % i, flat_masks[:, i].reshape((9, 64, -1)))


# In[255]:

flat_kernels = kernels.transpose((0, 2, 3, 1, 4))
print flat_kernels.shape
for i in xrange(32):
    imageio.mimsave('kernels.%d.gif' % i, flat_kernels[:, i].reshape((9, 5, -1)))


# In[256]:

flat_layers = (tf_layers/tf_layers.max(axis=(3, 4, 5), keepdims=True)).transpose((0, 2, 3, 1, 4, 5))
print flat_kernels.shape
for i in xrange(32):
    imageio.mimsave('layers.%d.gif' % i, flat_layers[:, i].reshape((9, 64, -1, 3)))


# In[257]:

flat_masked_layers = tf_layers * np.expand_dims(masks, 5)
flat_masked_layers = (flat_masked_layers/flat_masked_layers.max(axis=(3, 4, 5), keepdims=True)).transpose((0, 2, 3, 1, 4, 5))
print flat_masked_layers.shape
for i in xrange(32):
    imageio.mimsave('masked_layers.%d.gif' % i, flat_masked_layers[:, i].reshape((9, 64, -1, 3)))


def greyscale_to_rgb(a, time_dim=None):
    if time_dim == None:
        return np.transpose([a] * 3, (1, 2, 0))
    return np.transpose([a] * 3, (1, 2, 3, 0))


n_samples = 32
for i in xrange(n_samples):
    merged_mask_layer = np.stack([
            greyscale_to_rgb(flat_masks[:, i].reshape((9, 64, -1)), True),
            flat_layers[:, i].reshape((9, 64, -1, 3)),
            flat_masked_layers[:, i].reshape((9, 64, -1, 3))
        ], 1)
    imageio.mimsave('masks_n_layers.%d.gif' % i, merged_mask_layer.reshape((9, 3*64, 64*num_masks, 3)))
