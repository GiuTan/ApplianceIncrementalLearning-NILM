import numpy as np
import tensorflow as tf
from flags import *

tf.config.run_functions_eagerly(True) # Setting run_eagerly to True will help you debug that loop if anything goes wrong.

class LayerControl(tf.keras.callbacks.Callback):
    def __init__(self):
        pass
    def on_batch_end(self, batch, logs=None):
        # here you can get the model reference.
        tf.summary.histogram('arg_min', data= logs['arg_min'],step=batch)
        #tf.summary.histogram('grad_sum', data=logs['grad_sum'], step=batch)
        #tf.summary.histogram('arg_grad_max', data=logs['arg_grad_max'], step=batch)

def CRNN_block(x, kernel,drop_out,filters):
    conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(kernel, 1), strides=(1, 1), padding='same',
                                    kernel_initializer='glorot_uniform')(x)

    batch_norm_1 = tf.keras.layers.BatchNormalization()(conv_1, training=False)
    act_1 = tf.keras.layers.Activation('relu')(batch_norm_1)
    pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1))(act_1)
    drop_1 = tf.keras.layers.Dropout(drop_out)(pool_1)

    return drop_1


def CRNN_construction_final_n(window_size,initial_model, lr=0.0, classes=0, drop_out = 1.0, kernel = 1, num_layers=1, gru_units=1, flags=[True, True,True,True]):

    input_data = tf.keras.Input(shape=(window_size, 1))
    x = tf.keras.layers.Reshape((window_size,1,1))(input_data)

    for i in range(num_layers):
        filters = 2 ** (i+5)
        CRNN = CRNN_block(x, kernel=kernel, drop_out=drop_out, filters=filters)
        x = CRNN


    spec_x = tf.keras.layers.Reshape((x.shape[1], x.shape[3]))(x)
    bi_direct = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=gru_units,return_sequences=True))(spec_x)


    model_CRNN = tf.keras.Model(inputs=input_data, outputs=bi_direct,
                                name="CRNN")

    model_CRNN.set_weights(initial_model[:24])
    if more == 'one':
        instance_level = tf.keras.layers.Dense(units=classes+1)(model_CRNN.layers[-1].output)
    else:
        instance_level = tf.keras.layers.Dense(units=classes + 2)(model_CRNN.layers[-1].output)


    frame_level = tf.keras.layers.Activation('sigmoid', name = "strong_level")(instance_level)

    new_model_CRNN = tf.keras.Model(inputs=model_CRNN.inputs, outputs=frame_level,
                                name="CRNN")

    new_model_CRNN.layers[0].trainable = False
    new_model_CRNN.layers[1].trainable = False
    new_model_CRNN.layers[2].trainable = flags[0] #
    new_model_CRNN.layers[3].trainable = False
    new_model_CRNN.layers[4].trainable = False
    new_model_CRNN.layers[5].trainable = False
    new_model_CRNN.layers[6].trainable = False
    new_model_CRNN.layers[7].trainable = flags[1] #
    new_model_CRNN.layers[8].trainable = False
    new_model_CRNN.layers[9].trainable = False
    new_model_CRNN.layers[10].trainable = False
    new_model_CRNN.layers[11].trainable = False
    new_model_CRNN.layers[12].trainable = flags[2] #
    new_model_CRNN.layers[13].trainable = False
    new_model_CRNN.layers[14].trainable = False
    new_model_CRNN.layers[15].trainable = False
    new_model_CRNN.layers[16].trainable = False
    new_model_CRNN.layers[17].trainable = False
    new_model_CRNN.layers[18].trainable = flags[3] #
    new_model_CRNN.layers[19].trainable = True #


    return new_model_CRNN


@tf.function # inutile anzi dannoso perché mi da i gradienti none
def tape_grad(a,b,tape):
    grad = tape.gradient(a,b)
    return grad

class STU_TEACH_dyn(tf.keras.Model):

    def __init__(self, initial, final_x, final_1, final_2, final_3, final_4, classes ):
        super(STU_TEACH_dyn, self).__init__()
        self.final_x = final_x
        self.final_1 = final_1
        self.final_2 = final_2
        self.final_3 = final_3
        self.final_4 = final_4 # tanti modelli quanti layer vado a freezare singolarmente
        self.initial = initial
        self.classes = classes
        if more == 'one':
            self.new = 1
        else:
            self.new = 2

    def compile(self, final_optimizer, loss, loss_weights, F1_score):
        super(STU_TEACH_dyn, self).compile()
        self.final_optimizer = final_optimizer

        self.loss = loss
        self.loss_weights = loss_weights
        self.f1_score = F1_score

    def train_step(self,data):

        x = data[0]
        y = data[1]


        list_layers = [self.final_x.layers[2],self.final_x.layers[7], self.final_x.layers[12],self.final_x.layers[18]]  #
        # Train the student
        i = 0
        list_loss = []

        for flag in list_layers: # for each layer da freezare
            with tf.GradientTape() as tape1: # create the gradient
                self.final_x.load_weights('best_current_.h5') # load initial or best current model (nella prima epoca è il modello pre-trainato)

                predictions_final_x = self.final_x(x)
                predictions_initial = self.initial(x)
                KD_loss_x= self.loss["KD_loss"](predictions_initial, predictions_final_x[:, :, :self.classes]) # (predictions_initial[:,:,:1],predictions_final_x[:,:,:1]) #(predictions_initial, predictions_final_x[:, :, :self.classes])
                classification_loss_x = self.loss["final_loss"](y[:, :, -self.new:], predictions_final_x[:, :, -self.new:])
                sum_loss_x = self.loss_weights[1] * KD_loss_x + self.loss_weights[0] * classification_loss_x


                i = i + 1
                if i == 1:
                    flag.trainable = True
                    self.final_x.layers[7].trainable = False
                    self.final_x.layers[12].trainable = False
                    self.final_x.layers[18].trainable = False
                if i == 2:
                    self.final_x.layers[2].trainable = False
                    self.final_x.layers[12].trainable = False
                    self.final_x.layers[18].trainable = False
                    flag.trainable = True
                if i == 3:
                    self.final_x.layers[2].trainable = False
                    self.final_x.layers[7].trainable = False
                    self.final_x.layers[18].trainable = False
                    flag.trainable = True
                if i == 4:
                    self.final_x.layers[2].trainable = False
                    self.final_x.layers[7].trainable = False
                    self.final_x.layers[12].trainable = False
                    flag.trainable = True

                grads_x = tape1.gradient(sum_loss_x, self.final_x.trainable_weights)


            if i == 1:
		self.final_1.load_weights('best_current_.h5')
                self.final_optimizer.apply_gradients(zip(grads_x, self.final_1.trainable_weights))
                predictions_final_x = self.final_1(x)
                KD_loss_x = self.loss["KD_loss"](predictions_initial, predictions_final_x[:, :, :self.classes])  #(predictions_initial[:,:,:1],predictions_final_x[:,:,:1])
                classification_loss_x = self.loss["final_loss"](y[:, :, -self.new:], predictions_final_x[:, :, -self.new:])
                sum_loss_x = self.loss_weights[1] * KD_loss_x + self.loss_weights[0] * classification_loss_x
                list_loss.append(sum_loss_x.numpy())
            if i == 2:
		self.final_2.load_weights('best_current_.h5')
                self.final_optimizer.apply_gradients(zip(grads_x, self.final_2.trainable_weights))
                predictions_final_x = self.final_2(x)
                KD_loss_x = self.loss["KD_loss"](predictions_initial, predictions_final_x[:, :, :self.classes])  #(predictions_initial[:,:,:1],predictions_final_x[:,:,:1])
                classification_loss_x = self.loss["final_loss"](y[:, :, -self.new:], predictions_final_x[:, :, -self.new:])
                sum_loss_x = self.loss_weights[1] * KD_loss_x + self.loss_weights[0] * classification_loss_x
                list_loss.append(sum_loss_x.numpy())
            if i == 3:
		self.final_3.load_weights('best_current_.h5')
                self.final_optimizer.apply_gradients(zip(grads_x, self.final_3.trainable_weights))
                predictions_final_x = self.final_3(x)
                KD_loss_x = self.loss["KD_loss"](predictions_initial, predictions_final_x[:, :, :self.classes])  #(predictions_initial[:,:,:1],predictions_final_x[:,:,:1])
                classification_loss_x = self.loss["final_loss"](y[:, :, -self.new:], predictions_final_x[:, :, -self.new:])
                sum_loss_x = self.loss_weights[1] * KD_loss_x + self.loss_weights[0] * classification_loss_x
                list_loss.append(sum_loss_x.numpy())
            if i == 4:
		self.final_4.load_weights('best_current_.h5')
                self.final_optimizer.apply_gradients(zip(grads_x, self.final_4.trainable_weights))
                predictions_final_x = self.final_4(x)
                KD_loss_x = self.loss["KD_loss"](predictions_initial, predictions_final_x[:, :, :self.classes])  #(predictions_initial[:,:,:1],predictions_final_x[:,:,:1])
                classification_loss_x = self.loss["final_loss"](y[:, :, -self.new:], predictions_final_x[:, :, -self.new:])
                sum_loss_x = self.loss_weights[1] * KD_loss_x + self.loss_weights[0] * classification_loss_x
                list_loss.append(sum_loss_x.numpy())



        arg_max = np.argmin(list_loss)


        if round(arg_max) == 0 :
            self.final_1.save_weights('best_current_.h5')
        if round(arg_max) == 1:
            self.final_2.save_weights('best_current_.h5')
        if round(arg_max) == 2:
            self.final_3.save_weights('best_current_.h5')
        if round(arg_max) == 3:
            self.final_4.save_weights('best_current_.h5')



        return {"loss_model1": list_loss[0], "loss_model2": list_loss[1], "loss_model3": list_loss[2], "loss_model4": list_loss[3],
                "arg_min": arg_max, "loss_min": list_loss[arg_max]}
    def test_step(self, data):
        x = data[0]
        y = data[1]

        self.final_x.load_weights('best_current_.h5')
        predictions_final_x = self.final_x(x)

        f1_stu_x = self.f1_score(y,predictions_final_x)

        return {"f1_score": f1_stu_x }

    def call(self, inputs, *args, **kwargs):
        return self.final_x(inputs)