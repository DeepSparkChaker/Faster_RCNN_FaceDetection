# Import libraries
import keras
import Chaka_Faster_rcnn.common as common
import  Chaka_Faster_rcnn.layers as layers

# Get a VGG-16 model
def get_vgg_16_model(input=None):

    # Make sure that the input is okay
    input_shape = (None, None, 3)
    if input is None:
        input = keras.layers.Input(shape=input_shape)
    else:
        if not keras.backend.is_keras_tensor(input):
            input = keras.layers.Input(tensor=input, shape=input_shape)

    # Set backbone axis
    bn_axis = 3

    # Block 1
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    return x

# Get training models
def get_training_models(config=None, num_classes=2, weights_path=None):

    # Make sure that there is configurations
    if config == None:
        config = common.Config()

    # Calculate the number of anchors
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)

    # Create input layers
    img_input = keras.layers.Input(shape=(None, None, 3))
    roi_input = keras.layers.Input(shape=(None, 4))

    # Get a backbone model (VGG here, can be Resnet50, Inception, etc)
    backbone = get_vgg_16_model(img_input)

    # Create an rpn layer
    rpn = layers.rpn_layer(backbone, num_anchors)

    # Create a classifier
    classifier = layers.classifier_layer(backbone, roi_input, config.num_rois, nb_classes=num_classes)

    # Create models
    rpn_model = keras.models.Model(img_input, rpn[:2])
    classifier_model = keras.models.Model([img_input, roi_input], classifier)
    total_model = keras.models.Model([img_input, roi_input], rpn[:2] + classifier)

    # Load weights
    if weights_path != None:
        rpn_model.load_weights(weights_path, by_name=True)
        classifier_model.load_weights(weights_path, by_name=True)

    # Compile models
    rpn_model.compile(optimizer=keras.optimizers.Adam(lr=1e-5), loss=[common.rpn_loss_cls(num_anchors, config=config), common.rpn_loss_regr(num_anchors, config=config)])
    classifier_model.compile(optimizer=keras.optimizers.Adam(lr=1e-5), loss=[common.class_loss_cls(config=config), common.class_loss_regr(num_classes-1, config=config)], metrics={'dense_class_{}'.format(num_classes): 'accuracy'})
    total_model.compile(optimizer='sgd', loss='mae')

    # Return models
    return rpn_model, classifier_model, total_model

# Get inference models
def get_inference_models(config=None, num_classes=2, weights_path=None):

    # Make sure that there is configurations
    if config == None:
        config = common.Config()

    # Calculate the number of anchors
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)

    # Create input layers
    img_input = keras.layers.Input(shape=(None, None, 3))
    roi_input = keras.layers.Input(shape=(config.num_rois, 4))
    feature_map_input = keras.layers.Input(shape=(None, None, 512))

    # Get a backbone model (VGG here, can be Resnet50, Inception, etc)
    backbone = get_vgg_16_model(img_input)

    # Create an rpn layer
    rpn = layers.rpn_layer(backbone, num_anchors)

    # Create a classifier
    classifier = layers.classifier_layer(feature_map_input, roi_input, config.num_rois, nb_classes=num_classes)

    # Create models
    rpn_model = keras.models.Model(img_input, rpn)
    classifier_only_model = keras.models.Model([feature_map_input, roi_input], classifier)
    classifier_model = keras.models.Model([feature_map_input, roi_input], classifier)

    # Load weights
    rpn_model.load_weights(weights_path, by_name=True)
    classifier_model.load_weights(weights_path, by_name=True)

    # Compile models
    rpn_model.compile(optimizer='sgd', loss='mse')
    classifier_model.compile(optimizer='sgd', loss='mse')

    # Return models
    return rpn_model, classifier_model, classifier_only_model