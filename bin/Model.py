from tensorflow import keras
from tensorflow.keras import layers
from settings import config


class ConvPool:

    def __init__(self):

        inputConfig = list([int(a) for a in config["convpool_model"]["input_shape"].split(',')])
        inputs = keras.Input(shape = inputConfig)
        convConfig = list([{"name":n.strip(), "filters": int(f), "kernel_size" : int(k), "activation": a.strip()} for n,f,k,a in zip(
                                                    config["convpool_model"]["convolution_layer_names"].split(','),
                                                    config["convpool_model"]["convolution_filters"].split(','),
                                                    config["convpool_model"]["convolution_kernel_sizes"].split(','),
                                                    config["convpool_model"]["convolution_activation"].split(','))])

        mpConfig = list([{"name":n.strip(), "pool_size": int(p)} for n,p in zip(
                                                    config["convpool_model"]["maxpooling_layer_names"].split(','),
                                                    config["convpool_model"]["maxpooling_pool_size"].split(','))])

        m = layers.Conv2D(filters = convConfig[0]['filters'], kernel_size = convConfig[0]['kernel_size'], activation = convConfig[0]['activation'])(inputs)
        for conv, maxPool in zip(convConfig[1:], mpConfig):

            m = layers.MaxPooling2D(pool_size = maxPool['pool_size'])(m)
            m = layers.Conv2D(filters = conv['filters'] , kernel_size = conv['kernel_size'], activation = conv['activation'])(m)
        m = layers.Flatten()(m)
        outputs = layers.Dense(config["convpool_model"]["output_layer"], activation = config["convpool_model"]["output_activation"])(m)
        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def summary(self):
        return self.model.summary()

    def train(self, data):
        self.model.compile(
            optimizer = config["convpool_model"]["training_optimizer"],
            loss = config["convpool_model"]["training_loss_function"],
            metrics = config["convpool_model"]["training_metrics"].split(','),
            )
        self.model.fit(
            data.train, 
            epochs = int(config["convpool_model"]["training_epochs"]),
            validation_data = data.validation)
