import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from network.metrics_losses import custom_f1_score, StatefullF1

class WeightAdjuster_TS(tf.keras.callbacks.Callback):
  def __init__(self, weights: float):
    """
    Args:
    weights (list): list of loss weights
    """
    self.gamma = weights



  def on_epoch_end(self, epoch, logs=None):
      int_strong = np.log10(logs['KD_loss'])
      int_weak = np.log10(logs['weak_classification_loss'])
      int_loss = int_weak - int_strong
      int_loss2 =  10 ** (- int_loss)
      # Updated loss weights
      K.set_value(self.gamma, int_loss2)
      print(int_loss)
      tf.summary.scalar('balancing_factor', data=int_loss, step=epoch)


# class WeightAdjuster(tf.keras.callbacks.Callback):
#     def __init__(self, weights: list):
#         """
#         Args:
#         weights (list): list of loss weights
#         """
#         self.alpha = weights[0]
#         self.beta = weights[1]
#
#     def on_epoch_begin(self, epoch, logs=None):
#         x = epoch / 80
#         val_ = np.exp(-5.0 * (1.0 - x) ** 2.0)
#         val = tf.cast(val_, tf.float32)
#         new_beta = 50 * val
#         # Updated loss weights
#         K.set_value(self.beta, new_beta)
#         print(new_beta)
#         print(self.beta)
#
#         tf.summary.scalar('weight_unsup', data=new_beta, step=epoch)

class PredictionControl(tf.keras.callbacks.Callback):
    def __init__(self, train_data, val_data):

        self.x_train= train_data
        #self.x_val = val_data

    def on_epoch_end(self, epoch, logs=None):
        # here you can get the model reference.
        pred = self.model.predict(self.x_train)
        tf.summary.histogram('soft_student_train_preds', data=pred[0], step=epoch)



def loss_fn_sup(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)

    new_true = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32))
    new_pred = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32))

    loss = K.binary_crossentropy(new_true, new_pred)

    return tf.reduce_mean(loss)

def loss_fn_unsup(y_true, y_pred):

        #y_true = tf.cast(y_true, tf.float32)
        #new_true = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32))
        #new_pred = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32))
        loss = K.mean(K.sum(K.square(y_true - y_pred)))
        return loss

class STU_TEACH(tf.keras.Model):
    def __init__(self, initial, final, classes ):
        super(STU_TEACH, self).__init__()
        self.final = final
        self.initial = initial
        self.classes = classes
        self.new = 1

    def compile(self, final_optimizer, loss, loss_weights, F1_score):
        super(STU_TEACH, self).compile()
        self.final_optimizer = final_optimizer

        self.loss = loss
        self.loss_weights = loss_weights
        self.f1_score = F1_score

    def train_step(self,data):

        x = data[0]
        y = data[1]


        # Train the student
        with tf.GradientTape() as tape:
            predictions_final = self.final(x)
            predictions_initial = self.initial(x)

            KD_loss = self.loss["KD_loss"](predictions_initial[:,:,:1],predictions_final[:,:,:1])
            classification_loss = self.loss["final_loss"](y[:,:,-self.new:],predictions_final[:,:,-self.new:])
            sum_loss = self.loss_weights[1] * KD_loss + self.loss_weights[0] * classification_loss
            grads = tape.gradient(sum_loss, self.final.trainable_weights)


        self.final_optimizer.apply_gradients(zip(grads, self.final.trainable_weights))

        return {"sum_loss": sum_loss,"new_class_loss": classification_loss,"KD_loss":KD_loss} #

    def test_step(self, data):
        x = data[0]
        y = data[1]#[0]
        #y_w = data[1][2]

        predictions_final = self.final(x)
        predictions_initial = self.initial(x)

        KD_loss = self.loss["KD_loss"](predictions_initial[:,:,:1],predictions_final[:,:,:1]) # solo sul kettle

        classification_loss = self.loss["final_loss"](y[:,:,-self.new:],predictions_final[:,:,-self.new:])
        #strong_classification_loss = self.loss["student_loss"](y, predictions_stu[1])

        sum_loss = self.loss_weights[1] *  KD_loss + self.loss_weights[0] * classification_loss
        f1_stu = self.f1_score(y,predictions_final)

        return {"sum_loss": sum_loss, "final_loss": classification_loss, "KD_loss": KD_loss, "F1_score":f1_stu} #

    def call(self, inputs, *args, **kwargs):
        return self.final(inputs)








