from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Dense, Dropout


class KerasBuilder(object):
    """
    Keras model definition and compilation functions.
    """
    def __init__(self):
        pass

    @staticmethod
    def build_dense_model(input_size=None, input_layer_size=32, hidden_layer_size=32,
                          output_size=1, n_hidden_layers=2, activation_function='relu',
                          output_activation='linear', batch_normalization=False,
                          dropout=None, optimizer='adam', loss='mean_squared_error',
                          metrics=None):
        """
        Compiles and returns a Keras sequential model using dense fully-connected layers.

        Parameters
        ----------
        input_size : int, optional, default None
            Dimensionality of the input.

        input_layer_size : int, optional, default 32
            Input layer size.

        hidden_layer_size : int, optional, default 32
            Hidden layer size.

        output_size : int, optional, default 1
            Output layer size.

        n_hidden_layers : int, optional, default 2
            Number of hidden layers.

        activation_function : string, optional, default 'relu'
            Layer activation function.

        output_activation : string, optional, default 'linear'
            Output activation function.

        batch_normalization : boolean, optional, default False
            Whether or not to use batch normalization.

        dropout : int, optional, default None
            Amount of dropout per layer.

        optimizer : string, optional, default 'adam'
            Optimization method.

        loss : string, optional, default 'mean_squared_error'
            Loss function.

        metrics : list, optional, default none
            Evaluation metrics.
        """
        model = Sequential()

        # Input layer
        model.add(Dense(input_layer_size, input_dim=input_size))
        if batch_normalization:
            model.add(BatchNormalization())
        model.add(Activation(activation_function))
        if dropout:
            model.add(Dropout(dropout))

        # Hidden layers
        for i in range(n_hidden_layers):
            model.add(Dense(hidden_layer_size))
            if batch_normalization:
                model.add(BatchNormalization())
            model.add(Activation(activation_function))
            if dropout:
                model.add(Dropout(dropout))

        # Output layer
        model.add(Dense(output_size))
        model.add(Activation(output_activation))

        # Compile the model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model
