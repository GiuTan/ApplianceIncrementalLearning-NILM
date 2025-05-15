from network.CRNN import *
from utils_func import *
import random as python_random
import network.metrics_losses
import os
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from metrics import *
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
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

os.environ["CUDA_VISIBLE_DEVICES"]="7"

file_agg_path =  '/raid/users/eprincipi/KD_agg_UKDALE/'
file_labels_path = '/raid/users/eprincipi/KD_labels_UKDALE/'

type_= '_KE_WM_MW_' + dataset + '_' + house
print(type_)

X_val = np.load(file_agg_path + 'new_X_val.npy')
Y_val = np.load(file_labels_path + 'new_Y_val.npy')
X_refit = np.load('/raid/users/eprincipi/KD_agg_REFIT/new2_X_test.npy')
Y_refit = np.load('/raid/users/eprincipi/KD_labels_REFIT/new2_Y_test.npy')
X_train = np.load(file_agg_path + 'new_X_train.npy')
Y_train = np.load(file_labels_path + 'new_Y_train.npy')

Y_refit = np.delete(Y_refit, [3, 4, 5], 2)
Y_refit = Y_refit[:, :, [0, 2, 1]]
Y_val = np.delete(Y_val, [3, 4, 5], 2)
Y_val = Y_val[:, :, [0, 2, 1]]
Y_train = np.delete(Y_train, [3, 4, 5], 2)
Y_train = Y_train[:, :, [0, 2, 1]]
x_train = X_train
y_strong_train = Y_train

if house == 'H4':
        X_refit = X_refit[c:d]
        Y_refit = Y_refit[c:d]

        X_test = X_refit[180:]
        Y_test = Y_refit[180:]


if house == 'H2':
    
    X_refit = X_refit[a:b]
    Y_refit = Y_refit[a:b]

    if classes == 2 or classes == 3:

        X_test = X_refit[180:]
        Y_test = Y_refit[180:]
    if classes == 4:
        X_test = X_refit[180*2:]
        Y_test = Y_refit[180*2:]


# # Aggregate Standardization #
# if dataset == 'REFIT':
#     train_mean = 481.58
#     train_std = 657.87
# else:
#     train_mean = 279.42
#     train_std = 393.94


x_train = np.concatenate([x_train,X_refit[:180]])
Y_train_dw = Y_refit[:180]
y_strong_train[:, :, 1:2] = -1 # escludo labels kettle e wash
Y_train_dw[:,:, 0:1] = -1 # escludo labels dish
Y_train_dw[:,:, -1:] = -1
y_strong_train = np.concatenate([y_strong_train,Y_train_dw], axis=0)
train_mean = np.mean(x_train)
train_std = np.std(x_train)
print("Mean train")
print(train_mean)
print("Std train")
print(train_std)


x_train = standardize_data(x_train, train_mean, train_std)
X_val = standardize_data(X_val, train_mean, train_std)
X_test = standardize_data(X_test , train_mean, train_std)
Y_test= np.where(Y_test >0.5, 1, 0)
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

initial_CRNN = CRNN_construction(window_size,weight, lr=lr, classes=classes, drop_out=drop, kernel = kernel, num_layers=num_layers, gru_units=gru_units, cs=None,strong_weak_flag=True, temperature=1.0)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', mode='max',
                                                          patience=patience, restore_best_weights=True)

log_dir_ = 'logs_CRNN'  + datetime.now().strftime("%Y%m%d-%H%M%S") + type_
tensorboard = TensorBoard(log_dir=log_dir_)
file_writer = tf.summary.create_file_writer(log_dir_ + "/metrics")
file_writer.set_as_default()



history_ = initial_CRNN.fit(x=x_train, y=y_strong_train, shuffle=True, epochs=1000, batch_size=batch_size, validation_data=(X_test, Y_test), callbacks=[early_stop, tensorboard], verbose=1)
#initial_CRNN.save_weights('/raid/users/eprincipi/CL_nilm/CRNN_model_' + type_ + '.h5')

#initial_CRNN.load_weights('/raid/users/eprincipi/CL_nilm/CRNN_model_' + type_ + '.h5')

X_val = X_test
Y_val = Y_test
output_strong = initial_CRNN.predict(x=X_val)
output_strong_test = initial_CRNN.predict(x=X_test)

print(Y_val.shape)
print(output_strong.shape)


shape = output_strong.shape[0] * output_strong.shape[1]
shape_test = output_strong_test.shape[0] * output_strong_test.shape[1]

Y_val = Y_val.reshape(shape, classes)
Y_test = Y_test.reshape(shape_test, classes)

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
print(multilabel_confusion_matrix(Y_test, output_strong_test))
print(classification_report(Y_test, output_strong_test))
