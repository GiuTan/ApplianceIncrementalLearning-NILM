from network.CRNN_custom import *
from network.CRNN import *
from utils_func import *
import random as python_random
import network.metrics_losses
import os
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from metrics import *
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
from network.pruning import *
from flags import *


random.seed(123)
np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(1234)
tf.experimental.numpy.random.seed(1234)
os.environ['PYTHONHASHSEED'] = str(123)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

os.environ["CUDA_VISIBLE_DEVICES"]="6"

file_agg_path =  '../agg_UKDALE/'
file_labels_path = '../labels_UKDALE/'


type_= '_KE_WM_MW_' + dataset
        

X_val = np.load(file_agg_path + 'X_val.npy')
Y_val = np.load(file_labels_path + 'Y_val.npy')
X_test_r = np.load('../agg_REFIT/X.npy')
Y_test_r = np.load('../labels_REFIT/Y.npy')
X_train = np.load(file_agg_path + 'X_train.npy')
Y_train = np.load(file_labels_path + 'Y_train.npy')

Y_test_r = np.delete(Y_test_r, [1, 3, 4, 5], 2)
Y_val = np.delete(Y_val, [1, 3, 4, 5], 2)
Y_train = np.delete(Y_train, [1, 3, 4, 5], 2)
x_train = X_train
y_strong_train = Y_train

if house == 'H4':
        X_test_r = X_test_r[c:d]
        Y_test_r = Y_test_r[c:d]

        X_test_r = X_test_r[180:]
        Y_test_r = Y_test_r[180:]


if house == 'H2':
    
    X_test_r = X_test_r[a:b]
    Y_test_r = Y_test_r[a:b]

    X_test_r = X_test_r[180:]
    Y_test_r = Y_test_r[180:]


# Aggregate Standardization #
if dataset == 'REFIT':
    train_mean = 481.58
    train_std = 657.87
else:
    train_mean = 279.42
    train_std = 393.94

print("Mean train")
print(train_mean)
print("Std train")
print(train_std)

x_train = standardize_data(x_train, train_mean, train_std)
X_val = standardize_data(X_val, train_mean, train_std)
X_test_r = standardize_data(X_test_r , train_mean, train_std)
Y_test_r = np.where(Y_test_r >0.5, 1, 0)
Y_val = np.where(Y_val > 0.5, 1, 0)
y_strong_train =  np.where(y_strong_train > 0.5, 1, 0)

window_size = 2550
drop = 0.1
kernel = 5
num_layers = 3
gru_units = 64
lr = 0.002
drop_out = drop
weight= 1e-2
gamma = K.variable(1.0)
weight_dyn = WeightAdjuster(weights=gamma)
initial_CRNN = CRNN_construction(window_size,weight, lr=lr, classes=classes, drop_out=drop, kernel = kernel, num_layers=num_layers, gru_units=gru_units, cs=None,strong_weak_flag=True, temperature=1.0)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', mode='max',
                                                          patience=patience, restore_best_weights=True)

log_dir_ = 'logs_CRNN'  + datetime.now().strftime("%Y%m%d-%H%M%S") + type_
tensorboard = TensorBoard(log_dir=log_dir_)
file_writer = tf.summary.create_file_writer(log_dir_ + "/metrics")
file_writer.set_as_default()


history_ = initial_CRNN.fit(x=x_train, y=y_strong_train, shuffle=True, epochs=1000, batch_size=batch_size,
                                validation_data=(X_val, Y_val), callbacks=[early_stop, tensorboard], verbose=1)
initial_CRNN.save_weights('../CRNN_model_' + type_ + '.h5')

X_val = X_test_r
Y_val = Y_test_r
output_strong = initial_CRNN.predict(x=X_val)
output_strong_test = initial_CRNN.predict(x=X_test_r)

print(Y_val.shape)
print(output_strong.shape)


shape = output_strong.shape[0] * output_strong.shape[1]
shape_test = output_strong_test.shape[0] * output_strong_test.shape[1]

Y_val = Y_val.reshape(shape, classes)
Y_test_r = Y_test_r.reshape(shape_test, classes)

                # Y_train = y_strong_train.reshape(shape_train, classes)
output_strong = output_strong.reshape(shape, classes)
output_strong_test = output_strong_test.reshape(shape_test, classes)

if thres_strong == 0:
                    thres_strong = thres_analysis(Y_val, output_strong,classes)


print(Y_val.shape)
print(output_strong.shape)
assert (Y_val.shape == output_strong.shape)

print(thres_strong)


output_strong_test = app_binarization_strong(output_strong_test, thres_strong, classes)
output_strong = app_binarization_strong(output_strong, thres_strong, classes)

print("Test")
print(multilabel_confusion_matrix(Y_test_r, output_strong_test))
print(classification_report(Y_test_r, output_strong_test))
