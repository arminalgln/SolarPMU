# =============================================================================
# class for model
# =============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM, LeakyReLU
from tensorflow.keras import Model
import numpy as np
# import keras
# from keras import layers
# from keras.layers import


class PMUF:
    """
    This is a class for forecasting solar irrediation    
    """

    def __init__(self, feature_numbers, output_resolution):
        self.feature_numbers = feature_numbers
        # self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.model = self.__make_model()

    # private funciton to make model
    def __make_model(self):
        model = keras.Sequential(
            [  # keras.Input(shape=(self.resolution, self.inp_num,)),
                LSTM(32,  name='lstm_1', return_sequences=True),
                LeakyReLU(0.2),
                LSTM(64, name='lstm_2'),
                LeakyReLU(0.2),
                Dense(512, name='lstm_2', activation='relu'),
                Dense(self.output_resolution, activation='sigmoid', name='output'),
            ])
        return model

    def opt_ls_mtr(self, **kwarg):
        """

        Parameters
        ----------
        **kwarg : string
            optimizer, loss and metric.

        Returns
        -------
        None.
        :param kwarg:

        """
        opt, ls, mtr = kwarg['optimizer'], kwarg['loss'], kwarg['metric']
        self.model.compile(
            optimizer=tf.keras.optimizers.get(opt),
            loss=tf.keras.losses.get(ls),
            metrics=[tf.keras.metrics.get(mtr)],
        )

    def train(self, inp, out, **kwarg):
        """

        Parameters
        ----------
        inp : matrix
            features that can be used for train, dev and test dataset.
        out : vector
            solar irrediation.
        **kwarg : integer
            batch and epoch.

        Returns
        -------
        Training calculation based on the **kwarg numbers

        """

        batch, epoch = kwarg['batch'], kwarg['epoch']
        # time_len,feature_num=kwarg['time_len'],kwarg['feature_num']

        # inp = inp.reshape((inp.shape[0], self.input_resolution, self.feature_numbers))
        # model_input=layers.Input((time_len,feature_num))
        # print(inp.shape)
        self.model.fit(
            inp,
            out,
            batch_size=batch,
            epochs=epoch
        )

    def solar_eval(self, inp, out):
        """

        Parameters
        ----------
        inp : matrix
            features for evaluation such as training, dev or test set.
        out : matrix
            True values to evaluate the predicted values.

        Returns
        -------
        TYPE
            evaluation.

        """
        # inp = inp.reshape((inp.shape[0], self.input_resolution, self.feature_numbers))

        return self.model.evaluate(inp, out)

    def solar_predict(self, inp):
        """
        
        Parameters
        ----------
        x_pred : matrix
            features for prediction.

        Returns
        -------
        vector
            predicted solar irrediation.

        """
        # inp = inp.reshape((inp.shape[0], self.input_resolution, self.feature_numbers))

        return self.model.predict(inp)

