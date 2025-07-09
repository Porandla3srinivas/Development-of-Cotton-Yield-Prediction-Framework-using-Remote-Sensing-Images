import tensorflow as tf
from keras import layers, models
from keras.layers import Attention
import numpy as np
from Evaluation import evaluation

import torch
import torchvision.transforms as transforms
from PIL import Image
# from vit_pytorch import ViT


def Model_Vision_Transformer(image, Activation_Function):
    # Load pre-trained Vision Transformer model
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1
    )


    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    # Extract features using the Vision Transformer model
    with torch.no_grad():
        features = model(input_tensor)

    return features


def channel_shuffle(x, groups):
    _, w, h, c = x.get_shape().as_list()
    channels_per_group = c // groups

    x = tf.reshape(x, [-1, w, h, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, w, h, c])
    return x


def shuffle_unit(inputs, in_channels, out_channels, groups):
    shortcut = inputs

    # Grouped convolution
    branch_filters = out_channels // 4
    x = layers.Conv2D(branch_filters, (1, 1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = channel_shuffle(x, groups)
    x = layers.DepthwiseConv2D((3, 3), padding='same', strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)

    # Pointwise convolution
    x = layers.Conv2D(out_channels - in_channels, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Concatenate with shortcut
    if out_channels == in_channels:
        x += shortcut
    else:
        x = tf.concat([x, shortcut], axis=-1)

    # Final channel shuffle
    x = channel_shuffle(x, groups)
    return x


def create_shufflenet(input_shape, num_classes, sol):
    groups = 3
    inputs = layers.Input(input_shape)
    # Initial convolutional layer
    x = layers.Conv2D(24, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # ShuffleNet stages
    x = shuffle_stage(x, 24, 144, groups=groups, repeat=3)
    x = shuffle_stage(x, 144, 288, groups=groups, repeat=7)
    x = shuffle_stage(x, 288, 576, groups=groups, repeat=3)

    # Global average pooling and classifier
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(sol[0], activation='softmax')(x)

    model = models.Model(inputs, outputs, name='shufflenet')
    return model


def shuffle_stage(inputs, in_channels, out_channels, groups, repeat):
    x = shuffle_unit(inputs, in_channels, out_channels, groups)
    for _ in range(repeat - 1):
        x = shuffle_unit(x, out_channels, out_channels, groups)
    return x


def Model_ViT_SNetV2(Data, Target, opt, sol=None):
    feature = Model_Vision_Transformer(Data, Target)
    per = round(len(feature) * 0.75)
    train_data = feature[:per, :, :]
    train_target = Target[:per, :]
    test_data = feature[per:, :, :]
    test_target = Target[per:, :]

    if sol is None:
        sol = [5, 5, 50]
    input_shape = (224, 224, 3)
    num_classes = 1
    # Create ShuffleNet model
    shufflenet_model = create_shufflenet(input_shape, num_classes, sol)
    shufflenet_model.compile(loss='binary_crossentropy', metrics=['acc'])
    shufflenet_model.add(Attention())
    shufflenet_model.fit(train_data, train_target, steps_per_epoch=sol[2], epochs=sol[1])
    pred = np.round(shufflenet_model.predict(test_data)).astype('int')
    Eval = evaluation(pred, test_target)
    return Eval, pred
