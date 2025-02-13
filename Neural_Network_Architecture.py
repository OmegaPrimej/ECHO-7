from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2

Define the NeuralNetwork class
class NeuralNetwork:
    def __init__(self, config):
        self.config = config
        self.model = self.define_model()

    # Define the neural network architecture
    def define_model(self):
        model = Sequential()
        model.add(Dense(self.config['input_dim'], activation='relu', input_shape=(self.config['input_dim'],)))
        model.add(Dropout(self.config['dropout_rate']))

        # Add hidden layers
        for i in range(self.config['num_hidden_layers']):
            model.add(Dense(self.config['hidden_dim'], activation='relu'))
            model.add(Dropout(self.config['dropout_rate']))

        # Add output layer
        model.add(Dense(self.config['output_dim'], activation='sigmoid'))

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    # Get the model
    def get_model(self):
        return self. model
