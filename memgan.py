from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
from keras import optimizers
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Dense, Flatten, Reshape, Concatenate
from keras.layers import LeakyReLU, Layer
from keras.layers.merge import _Merge
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from functools import partial
from layers import Conv2DWscale
from keras.initializers import Initializer
from keras.initializers import Initializer
from keras.utils import multi_gpu_model

GRADIENT_PENALTY_WEIGHT = 10

LOD_FILTERS = {
    2: 512,
    3: 512,
    4: 512,
    5: 512,
    6: 256,
    7: 128,
    8: 64
}

# Replace with your favourite memorability estimator.
MemModel = load_model("MODEL/PATH", compile=False)
MemModel.trainable = False
MemModel.summary()


# # Equalised weight scaling
# def get_scaled_weight(shape):
#     # He's normal dynamic weight scaler
#     if len(shape) == 2:
#         fan_in = shape[0]
#     else:
#         receptive_field_size = np.prod(shape[:-2])
#         fan_in = shape[-2] * receptive_field_size
#
#     std = np.sqrt(2 / max(1., fan_in))
#
#     return K.random_normal(shape, mean=0, stddev=1, dtype=tf.float32) * std
    #return tf.get_variable("w", shape=shape, initializer=tf.initializers.random_normal(0, 1), dtype=tf.float32) * std


# class get_scaled_weight_2:
#     def __call__(self, shape, dtype=None):
#         return get_scaled_weight(shape)


class get_scaled_weight(Initializer):
    def __call__(self, shape, dtype=None):
        # He's normal dynamic weight scaler
        if len(shape) == 2:
            fan_in = shape[0]
        else:
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size

        std = np.sqrt(2 / max(1., fan_in))

        return K.random_normal(shape, mean=0, stddev=1, dtype=tf.float32) * std


class Sigmoid(Layer):
    """
    Convert sigmoid activation into a layer.
    Alpha is not used.
    """
    def __init__(self, alpha=0.3, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return K.tanh(inputs)

    def get_config(self):
        config = {}
        base_config = super(Sigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


initialization = get_scaled_weight()


def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight, discriminator):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""

    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    #gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients = K.gradients(discriminator.get_layer('discriminator').outputs[-1], averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


#Pixelwise feature vector normalization layer from "Progressive Growing of GANs" paper
class PixelNorm(Layer):
    def __init__(self, epsilon=1e-08, **kwargs):
        self.eps = epsilon
        super(PixelNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PixelNorm, self).build(input_shape)

    def call(self, x):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)


def MiniBatchStddev(x, group_size=4): #again position of channels matter!
    import tensorflow as tf
    group_size = tf.minimum(group_size, tf.shape(x)[0])# Minibatch must be divisible by (or smaller than) group_size.
    s = x.shape                                             # [NCHW]  Input shape.
    y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
    y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
    y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
    y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
    y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)    # [M111]  Take average over fmaps and pixels.
    y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
    y = tf.tile(y, [group_size, s[1], s[2], 1])             # [N1HW]  Replicate over group and pixels.
    return tf.concat([x, y], axis=-1)


def scale_layer(layer, scale_factor):
    '''
    Upscales layer (tensor) by the factor (int) where
    the tensor is [group, height, width, channels]
    '''
    # Import inside lambda to workaround keras stupidity. Honestly.
    import tensorflow as tf
    shapes = layer.get_shape().as_list()
    height = shapes[1]
    width = shapes[2]
    size = (int(scale_factor * height), int(scale_factor * width))
    scaled_layer = tf.image.resize_nearest_neighbor(layer, size)
    return scaled_layer


class MergeLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(MergeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MergeLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        alpha = self.alpha
        last_fully_trained = inputs[0]
        new_layer = inputs[1]
        # This code block should take advantage of broadcasting
        new_layer = (1 - alpha) * last_fully_trained + new_layer * alpha

        return K.tanh(new_layer)
        #return new_layer

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        alpha = K.get_value(self.alpha)
        config = {
            'alpha': alpha
        }
        base_config = super(MergeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        alpha = config['alpha']
        config['alpha'] = K.variable(alpha, name="alpha")
        return cls(**config)


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self, inputs):
        BATCH_SIZE = K.shape(inputs[0])[0]
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1], 3), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1], 0] = img[:, :, 0]
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1], 1] = img[:, :, 1]
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1], 2] = img[:, :, 2]
    return image


def build_generator(lod, alpha, previous_gen=None):
    # This is the initial generator
    if previous_gen is None:
        latent_input = Input((512, ))
        mem_code_input = Input((1,))

        concatenated_inputs = Concatenate()([latent_input, mem_code_input])

        model = Dense(512 * 4 * 4)(concatenated_inputs)
        model = Reshape((4, 4, 512))(model)
        model = Conv2DWscale(512, (4, 4), name=str(lod) + "_conv1", padding='same')(model)
        model = LeakyReLU(name=str(lod) +"_conv1_leakyrelu", alpha=0.2)(model)
        model = Conv2DWscale(512, (3, 3), name=str(lod) + "_conv2", padding='same')(model)
        model = LeakyReLU(name=str(lod) +"_conv2_leakyrelu", alpha=0.2)(model)
        model = PixelNorm(name=str(lod) + "_pixelnorm1")(model)
        model = Conv2DWscale(3, (1, 1), name="initial_toRGB")(model)
        model = LeakyReLU(name="initial_toRGB_leakyrelu", alpha=0.2)(model)

        model = Model([latent_input, mem_code_input], model)
        return model

    fnum = LOD_FILTERS[lod]

    # Upscale previous_gan output
    if lod > 3:
        previous_layer = previous_gen.get_layer(str(lod - 1) + "_pixelnorm2")
    else:
        previous_layer = previous_gen.layers[-1]
    prev_upscale = Lambda(scale_layer, name=str(lod) + "_upscale", arguments={'scale_factor': 2})(previous_layer.output)
    prev_toRGB = Conv2DWscale(3, (1, 1), padding='same', name=str(lod) + "_prev_toRGB")(prev_upscale)
    prev_toRGB = LeakyReLU(name=str(lod) +"_prev_toRGB_leakyrelu", alpha=0.2)(prev_toRGB)

    new_model = Conv2DWscale(fnum, (3, 3), name=str(lod) + "_conv1", padding='same')(prev_upscale)
    new_model = LeakyReLU(name=str(lod) +"_conv1_leakyrelu", alpha=0.2)(new_model)
    new_model = PixelNorm(name=str(lod) + "_pixelnorm2")(new_model)
    new_model = Conv2DWscale(fnum, (3, 3), name=str(lod) + "_conv2", padding='same')(new_model)
    new_model = LeakyReLU(name=str(lod) +"_conv2_leakyrelu", alpha=0.2)(new_model)
    new_model = PixelNorm(name=str(lod) + "_pixelnorm3")(new_model)

    new_toRGB = Conv2DWscale(3, (1, 1), padding='same', name=str(lod) + "_toRGB")(new_model)
    new_toRGB = LeakyReLU(name=str(lod) + "_toRGB_leakyrelu", alpha=0.2)(new_toRGB)

    #new_merge_layer = Lambda(smoothly_merge_last_layer, name=str(lod)+"_merge_layer", arguments={'alpha': alpha})([prev_toRGB, new_toRGB])
    new_merge_layer = MergeLayer(alpha, name=str(lod)+"_merge_layer")([prev_toRGB, new_toRGB])

    return Model(previous_gen.input, new_merge_layer)


def build_discriminator(lod, alpha, previous_disc=None):
    if previous_disc is None:
        input_image = Input((4, 4, 3), name="initial_input")
        model = Conv2DWscale(512, (1, 1), padding='same', name="initial_fromRGB")(input_image)
        model = LeakyReLU(name=str(lod) +"initial_fromRGB_leakyrelu", alpha=0.2)(model)
        model = Lambda(MiniBatchStddev, name="minibatch_stddev")(model)
        model = Conv2DWscale(512, (3, 3), name=str(lod) + "_conv1", padding='same')(model)
        model = LeakyReLU(name=str(lod) +"_conv1_leakyrelu", alpha=0.2)(model)
        model = Conv2DWscale(512, (4, 4), name=str(lod) + "_conv2", padding='same')(model)
        model = LeakyReLU(name=str(lod) +"_conv2_leakyrelu", alpha=0.2)(model)
        model = Flatten()(model)
        output = Dense(1)(model)
        return Model(input_image, output)

    fnum = LOD_FILTERS[lod]
    fnum2 = fnum
    # Change this when layers start reducing filter numbers.
    if 2 < fnum < LOD_FILTERS[lod - 1]:
        fnum2 = LOD_FILTERS[lod - 1]

    input_image = Input((2**lod, 2**lod, 3), name=str(lod) + "_new_input")
    #prev_downsample = Lambda(scale_layer, name=str(lod) + "_downscale", arguments={'scale_factor': 0.5})(input_image)
    prev_downsample = AveragePooling2D((2, 2), name=str(lod) + "_downscale")(input_image)
    prev_fromRGB = Conv2DWscale(fnum2, (1, 1), padding='same', name=str(lod) + "_prev_fromRGB")(prev_downsample)
    prev_fromRGB = LeakyReLU()(prev_fromRGB)

    # New Block
    new_model = Conv2DWscale(fnum, (1, 1), padding='same', name=str(lod) + "_new_fromRGB")(input_image)
    new_model = LeakyReLU(name=str(lod) +"_new_fromRGB_leakyrelu", alpha=0.2)(new_model)

    new_model = Conv2DWscale(fnum, (3, 3), name=str(lod) + "_conv1", padding='same')(new_model)
    new_model = LeakyReLU(name=str(lod) +"_conv1_leakyrelu", alpha=0.2)(new_model)
    new_model = Conv2DWscale(fnum2, (3, 3), name=str(lod) + "_conv2", padding='same')(new_model)
    new_model = LeakyReLU(name=str(lod) +"_conv2_leakyrelu", alpha=0.2)(new_model)

    #downsample = Lambda(scale_layer, name=str(lod) + "_new_downscale", arguments={'scale_factor': 0.5})(new_model)
    downsample = AveragePooling2D((2, 2), name=str(lod) + "_new_downscale")(new_model)

    new_merge_layer = MergeLayer(alpha, name=str(lod) + "_merge_layer")([prev_fromRGB, downsample])

    if lod > 3:
        new_merge_layer = previous_disc.get_layer(str(lod - 1) + "_conv1")(new_merge_layer)
        new_merge_layer = previous_disc.get_layer(str(lod - 1) + "_conv1_leakyrelu")(new_merge_layer)
        new_merge_layer = previous_disc.get_layer(str(lod - 1) + "_conv2")(new_merge_layer)
        new_merge_layer = previous_disc.get_layer(str(lod - 1) + "_conv2_leakyrelu")(new_merge_layer)
        new_merge_layer = previous_disc.get_layer(str(lod - 1) + "_new_downscale")(new_merge_layer)
        if lod == 4:
            new_merge_layer = previous_disc.get_layer("minibatch_stddev")(new_merge_layer)
        if lod >= 6:
            for x in range((-6 * (lod - 3)) + (lod - 5), 0):
                new_merge_layer = previous_disc.layers[x](new_merge_layer)
        else:
            for x in range(-6 * (lod - 3), 0):
                new_merge_layer = previous_disc.layers[x](new_merge_layer)
    else:
        for x in range(3, len(previous_disc.layers)):
            new_merge_layer = previous_disc.layers[x](new_merge_layer)
    #
    # if lod == 3:
    #     # First modified structure
    #     for x in range(2, len(previous_disc.layers)):
    #         new_merge_layer = previous_disc.layers[x](new_merge_layer)

    return Model(input_image, new_merge_layer)


def average_layer(x):
    # Shape is [batch, h, w, c]
    shape = K.int_shape(x)
    modify = x[:, :, :, 1]
    modify = K.reshape(modify, (-1, shape[1]*shape[2]))
    modify = K.mean(modify, axis=-1)
    modify = K.expand_dims(modify)
    return modify


def resize_layer(x):
    resized_x = tf.image.resize_images(x, (224, 224))
    return resized_x


def memorability_aux(res):
    input = Input((res, res, 3))
    model = Lambda(resize_layer)(input)
    model = MemModel(model)
    model = Lambda(average_layer)(model)
    model = Model(input, model)
    return model


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def wasserstein_loss_reals(y_true, y_pred):
    # Add Epsilon term from ProGAN Paper
    return K.mean(y_true * y_pred) + 0.001 * K.square(y_pred)


def aux_loss(y_true, y_pred):
    #alpha = 10
    alpha = 5
    return alpha * (K.mean(K.square(y_pred - y_true), axis=-1))


adam_gen = optimizers.Adam(lr=0.0015, beta_1=0.0, beta_2=0.99, epsilon=10e-8)
adam_disc = optimizers.Adam(lr=0.0015, beta_1=0.0, beta_2=0.99, epsilon=10e-8)

# Parameters
#data_dir = "./test_dataset/"
#total_datapoints = 2

data_dir = "../kitchen_images_700/"
total_datapoints = 240000

img_height = 256
img_width = 256
img_channels = 3
LATENT_SPACE = 512

data_generator = ImageDataGenerator(
    data_format='channels_last',
    horizontal_flip=False)

flow_from_directory_params = {'target_size': (img_height, img_width),
                              'color_mode': 'grayscale' if img_channels == 1 else 'rgb',
                              'class_mode': None}


def set_alpha_variable_on_load(a, model, type="MODEL_NOT_DEFINED"):
    for layer in model.layers:
        if isinstance(layer, MergeLayer):
            print("Loading {} with alpha of: {}".format(type, K.get_value(layer.alpha)))
            layer.alpha = a


def construct_models(lod, a, G_old=None, D_old=None, load=False, epoch=0):
    if (G_old is None) and (not load):
        G = build_generator(lod, a)
        D = build_discriminator(lod, a)
        A = memorability_aux(2**lod)
    elif not load:
        D_old.trainable = True
        G = build_generator(lod, a, G_old)
        D = build_discriminator(lod, a, D_old)
        A = memorability_aux(2 ** lod)
    elif load:
        print("Loading from file.")
        from keras.utils.generic_utils import get_custom_objects

        get_custom_objects().update({'get_scaled_weight': get_scaled_weight})

        G = load_model('epoch_saves/' + str(epoch) + '_G.h5', custom_objects={"Conv2DWscale": Conv2DWscale,
                                                                                      "PixelNorm": PixelNorm,
                                                                                      "MergeLayer": MergeLayer,
                                                                                      "Sigmoid": Sigmoid})
        D = load_model('epoch_saves/' + str(epoch) + '_D.h5', custom_objects={"Conv2DWscale": Conv2DWscale,
                                                                                          "PixelNorm": PixelNorm,
                                                                                          "MergeLayer": MergeLayer,
                                                                                          "Sigmoid": Sigmoid})
        set_alpha_variable_on_load(a, G, "G")
        set_alpha_variable_on_load(a, D, "D")

        A = memorability_aux(2 ** lod)

    D.trainable = False
    A.trainable = False
    #critic = D(G.output)
    critic = [D(G.output), A(G.output)]
    GAN = Model(G.input, critic)
    GAN = multi_gpu_model(GAN, gpus=2)
    GAN.compile(optimizer=adam_gen, loss=[wasserstein_loss, aux_loss])

    D.trainable = True

    reals = Input((2**lod, 2**lod, 3))
    generated = Input((2**lod, 2**lod, 3))

    weighted_average = RandomWeightedAverage()([reals, generated])
    Discriminator = Model([reals, generated], [D(reals), D(generated), D(weighted_average)], name='discriminator')
    Discriminator = multi_gpu_model(Discriminator, gpus=2)

    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=weighted_average,
                              gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT,
                              discriminator=Discriminator)
    # Functions need names or Keras will throw an error
    partial_gp_loss.__name__ = 'gradient_penalty'

    Discriminator.compile(optimizer=adam_disc, loss=[wasserstein_loss_reals, wasserstein_loss, partial_gp_loss])

    return G, D, GAN, Discriminator


def get_image_batch(real_image_generator, BATCH_SIZE):
    img_batch = real_image_generator.next()

    # keras generators may generate an incomplete batch for the last batch in an epoch of data
    if len(img_batch) != BATCH_SIZE:
        img_batch = real_image_generator.next()

    assert img_batch.shape == (BATCH_SIZE, img_height, img_width, img_channels), img_batch.shape
    return img_batch


def resize_batch(batch, lod, alpha):
    out = []
    for image in batch:
        out_image = image.astype('uint8')
        out_image = Image.fromarray(out_image)
        # Interpolate between resolutions:
        if lod > 2:
            # Shrink image to previous lod
            prev_image = out_image.resize((2 ** (lod-1), 2 ** (lod-1)))
            # Resize the image back to current lod
            prev_image = prev_image.resize((2 ** lod, 2 ** lod))
            #Resize an image directly to current lod
            out_image = out_image.resize((2 ** lod, 2 ** lod))
            # Interpolate between the two using current alpha
            out_image = Image.blend(prev_image, out_image, alpha)
        else:
            out_image = out_image.resize((2 ** lod, 2 ** lod))
        out_image = np.asarray(out_image, dtype='float32')
        out_image = (out_image - 127.5) / 127.5
        out.append(out_image)
    return np.array(out)


# Initial setup and checkpointing
load_from_file = False
LOAD_EPOCH = 0
LOAD_LOD = 0
LOAD_ALPHA = 0.0
LOAD_CHECKPOINT_INDEX = 0

EPOCHS = 300
if not load_from_file:
    checkpoint_index = 0
    lod = 2
    current_epoch = 0
    a = K.variable(1.0, name="alpha")
    G, D, GAN, Discriminator = construct_models(lod, a)
    alpha = 1.0
    EPOCHS = [EPOCHS]
else:
    lod = LOAD_LOD
    alpha = LOAD_ALPHA
    checkpoint_index = LOAD_CHECKPOINT_INDEX
    a = K.variable(alpha, name="alpha")
    G, D, GAN, Discriminator = construct_models(lod, a, load=True, epoch=LOAD_EPOCH)
    EPOCHS = [LOAD_EPOCH, EPOCHS]

# Initial training stage of 10 epochs
# Grow GAN after 10 epochs, stabilise for 10 epochs

# Train the discriminator n times more than the generator.
n_critic = 1

grow_checkpoints = [20, 60, 100, 140, 180, 220, 260]
batch_sizes = [256, 128, 128, 64, 32, 16, 8]

# TEST DATA
#grow_checkpoints = [2, 4, 6]
#batch_sizes = [2, 2, 1, 1, 1, 1, 1]


def build_data_gen(batch_size):
    batch_size = {'batch_size': batch_size}
    params = dict(flow_from_directory_params.items() + batch_size.items())
    real_image_generator = data_generator.flow_from_directory(
        directory=data_dir,
        **params
    )

    return real_image_generator


real_image_generator = build_data_gen(batch_sizes[checkpoint_index])


# Training Loop
for epoch in range(*EPOCHS):
    # Compute batchsize for this epoch.
    batches_per_epoch = total_datapoints / batch_sizes[checkpoint_index]
    BATCH_SIZE = batch_sizes[checkpoint_index]
    print("Epoch: {} batch_size: {} batch_count: {} alpha: {} LOD: {}".format(epoch, BATCH_SIZE, batches_per_epoch, alpha, lod))

    if epoch % 10 == 0:
        if load_from_file and epoch == LOAD_EPOCH:
            print("No save as model just loaded.")
        else:
            print("Saving - LOD: {} Alpha: {}".format(lod, alpha))
            G.save('epoch_saves/' + str(epoch) + '_G.h5')
            D.save('epoch_saves/' + str(epoch) + '_D.h5')
            GAN.save('epoch_saves/' + str(epoch) + '_GAN.h5')
            Discriminator.save('epoch_saves/' + str(epoch) + '_discriminator.h5')

    if alpha < 1.0:
        # Fade in for n epochs: alpha = 1.0/n
        alpha += 0.025
        if alpha > 1.0:
            alpha = 1.0
        K.set_value(a, alpha)

    if epoch == grow_checkpoints[checkpoint_index]:
        if checkpoint_index < len(grow_checkpoints):
            if checkpoint_index != len(grow_checkpoints) - 1:
                checkpoint_index += 1
            lod += 1
            print("LOD: ", lod)
            alpha = 0.0
            K.set_value(a, alpha)
            G, D, GAN, Discriminator = construct_models(lod, a, G, D)
            print("Successfully grown GAN")

            real_image_generator = build_data_gen(batch_sizes[checkpoint_index])
            batches_per_epoch = total_datapoints / batch_sizes[checkpoint_index]
            BATCH_SIZE = batch_sizes[checkpoint_index]

            print("Recomputed batch sizes")

    for index in range(int(batches_per_epoch)):
        #print("Starting discriminator training loop")
        #mem_noise = np.random.normal(0, 60, size=(BATCH_SIZE, 1))
        mem_noise = np.random.normal(30, 30, size=(BATCH_SIZE, 1))
        for n in range(n_critic):
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, LATENT_SPACE))
            image_batch = get_image_batch(real_image_generator, BATCH_SIZE)
            image_batch = resize_batch(image_batch, lod, alpha)
            generated_images = G.predict([noise, mem_noise])

            #X = np.concatenate((image_batch, generated_images))
            y1 = np.random.uniform(0.8, 1.2, BATCH_SIZE)
            y2 = np.random.uniform(-0.8, -1.2, BATCH_SIZE)
            # y = np.concatenate((y1, y2))
            # d_loss = D.train_on_batch(X, y)


            dummy_y = np.zeros((BATCH_SIZE, 1))
            d_loss = Discriminator.train_on_batch([image_batch, generated_images], [y1, y2, mem_noise])
            # print("batch {} d_loss_total : {} d_loss_real: {}, d_loss_fake: {}, d_loss_GP: {}"
            #       .format(index, d_loss[0], d_loss[1], d_loss[2], d_loss[3]))

        dummy_mem = np.zeros((BATCH_SIZE, 1))
        noise = np.random.normal(size=(BATCH_SIZE, LATENT_SPACE))
        mem_noise = np.random.normal(30, 30, size=(BATCH_SIZE, 1))
        g_loss = GAN.train_on_batch([noise, mem_noise], [np.ones((BATCH_SIZE, 1)), dummy_mem])
        #print("batch {} g_loss : {}, g_aux: {}".format(index, g_loss[0], g_loss[1]))

    if epoch % 1 == 0:
        noise = np.random.normal(0, 1, size=(BATCH_SIZE, LATENT_SPACE))
        #mem_noise = np.random.normal(0, 60, size=(BATCH_SIZE, 1))
        mem_noise = abs(np.random.normal(30, 30, size=(BATCH_SIZE, 1)))
        #image_batch = get_image_batch(real_image_generator, BATCH_SIZE)
        #image_batch = resize_batch(image_batch, lod, alpha)
        generated_images = G.predict([noise, mem_noise])

        image = combine_images(generated_images)
        image = image * 127.5
        image = image + 127.5
        Image.fromarray(image.astype(np.uint8)).save(
            str("out_images/epoch") + "_" + str(epoch) + ".png")

    print("Epoch {} completed".format(epoch))

G.save('G.h5')
D.save('D.h5')
Discriminator.save('discriminator.h5')
GAN.save('GAN.h5')

