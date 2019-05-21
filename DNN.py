from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as k
import tensorflow as tf


class DNN:
    def __init__(self, state_size, action_size, learning_rate, layers, loss):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self._build_model(loss, layers)

    def _build_model(self, loss, layers):
        # Neural Net for Deep-Q learning Model
        model = Sequential()

        model.add(Dense(layers[0], input_dim=self.state_size, activation='linear', kernel_initializer='random_uniform', bias_initializer='zeros'))

        for i in range(1, len(layers)):
            model.add(Dense(layers[i], activation='linear', kernel_initializer='random_uniform', bias_initializer='zeros'))

        model.add(Dense(self.action_size, activation='linear', kernel_initializer='random_uniform', bias_initializer='zeros'))

        if loss == "huber":
            model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))
        else:
            model.compile(loss=loss, optimizer=Adam(lr=self.learning_rate))

        return model

    def predict(self, state):
        return self.model.predict(state)[0]

    def update_weights(self, other_model):
        self.model.set_weights(other_model.model.get_weights())

    """Huber loss for Q Learning
        References: https://en.wikipedia.org/wiki/Huber_loss
                    https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
        """
    @staticmethod
    def _huber_loss(y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = k.abs(error) <= clip_delta

        squared_loss = 0.5 * k.square(error)
        quadratic_loss = 0.5 * k.square(clip_delta) + clip_delta * (k.abs(error) - clip_delta)

        return k.mean(tf.where(cond, squared_loss, quadratic_loss))

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
