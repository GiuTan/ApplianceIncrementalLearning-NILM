from network.CRNN import *
from utils_func import *
import random as python_random
from network.TEACH_STU import *
import network.metrics_losses
import os
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from metrics import *
from flags import *
from network.dynamic_finetuning import *


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


X = np.load('../agg_REFIT/agg.npy')
Y = np.load('../labels_REFIT/labels.npy')

if classes == 2 and more == 'one':
    model_ = '_M2+1_'
    
else:
    if classes == 2 and more == 'two':
        model_ = '_M2+2_'
        
    else:
        model_ = '_M2+1+1_'
        

type_ = model_ + str(house) + '_AIL_' + dataset + '_' + str(batch_size) + str(loss_weights) + 'PROVACODICE'


if house == 'H2':

    if classes == 2:

        X = X[a:b]
        Y = Y[a:b]
        X_train = X[:180]
        Y_train = Y[:180]
        X_test = X[180:]
        Y_test = Y[180:]

    if classes == 3: # 2 + 1

        X = X[a:b]
        Y = Y[a:b]
        X_train = X[180:180*2]
        Y_train = Y[180:180*2]
        X_test = X[180*2:]
        Y_test = Y[180*2:]

if house=='H4':

    if classes == 2:
        X = X[c:d]
        Y = Y[c:d]
        X_train = X[:180]
        Y_train = Y[:180]
        X_test = X[180:]
        Y_test = Y[180:]

    if classes == 3:  # 2 + 1

        X = X[c:d]
        Y = Y[c:d]
        X_train = X[180:180 * 2]
        Y_train = Y[180:180 * 2]
        X_test = X[180 * 2:]
        Y_test = Y[180 * 2:]

if house == 'H2':
        if classes == 2 and more == 'one':

                Y_test = np.delete(Y_test, [1, 4, 5], 2)  # elimino  microwave, toaster e washer dryer
                Y_train = np.delete(Y_train, [1, 4, 5], 2)
        else:
            if classes == 2 and more == 'two':
                Y_test = np.delete(Y_test, [1, 5], 2)  # elimino solo microwave e washer dryer
                Y_train = np.delete(Y_train, [1, 5], 2)
            else:
                Y_test = np.delete(Y_test, [1, 5], 2)  # elimino solo microwave e washer dryer
                Y_train = np.delete(Y_train, [1, 5], 2)

if house == 'H4':

        Y_test = np.delete(Y_test, [3, 4, 5], 2)  # elimino  dishwasher, toaster e washer dryer
        Y_test = Y_test[:, :, [0, 2, 1]] # scambio posizione mw e wm [ke wm mw]
        Y_train = np.delete(Y_train, [3, 4, 5], 2)
        Y_train = Y_train[:, :, [0, 2, 1]]

# Aggregate Standardization #
train_mean = 279.42
train_std = 393.94
print("Mean train")
print(train_mean)
print("Std train")
print(train_std)

X_val = X_test
Y_val = Y_test

X_train = standardize_data(X_train, train_mean, train_std)
X_val = standardize_data(X_val, train_mean, train_std)
X_test = standardize_data(X_test , train_mean, train_std)

Y_train = np.where(Y_train > 0.5, 1, 0)
Y_test = np.where(Y_test >0.5, 1, 0)
Y_val = np.where(Y_val >0.5, 1, 0)

window_size = 2550
drop = 0.1
kernel = 5
num_layers = 3
gru_units = 64
lr = 0.002
drop_out = drop
weight= 1e-2
gamma = K.variable(1.0)

initial_CRNN = CRNN_construction(window_size,weight, lr=lr, classes=classes, drop_out=drop, kernel = kernel, num_layers=num_layers, gru_units=gru_units, cs=None,strong_weak_flag=True, temperature=1.0)
initial_CRNN.load_weights(initial_model_path)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', mode='max',
                                                          patience=patience, restore_best_weights=True)

if approach == 'LwF':
    batch_size_ = batch_size
    final_CRNN = CRNN_construction_final(window_size, initial_model = initial_CRNN.get_weights(), lr=lr, classes=(classes), drop_out=drop, kernel = kernel, num_layers=num_layers, gru_units=gru_units)
    final_CRNN.summary()
    MODEL = STU_TEACH(initial_CRNN, final_CRNN, classes)

if approach == 'AIL':
    batch_size_ = batch_size_dyn
    final_CRNN_x = CRNN_construction_final_n(window_size, initial_model=initial_CRNN.get_weights(), lr=lr,
                                             classes=(classes), drop_out=drop, kernel=kernel, num_layers=num_layers,
                                             gru_units=gru_units, flags=[False, False, False, False])
    final_CRNN_1 = CRNN_construction_final_n(window_size, initial_model=initial_CRNN.get_weights(), lr=lr,
                                             classes=(classes), drop_out=drop, kernel=kernel, num_layers=num_layers,
                                             gru_units=gru_units, flags=[True, False, False, False])
    final_CRNN_2 = CRNN_construction_final_n(window_size, initial_model=initial_CRNN.get_weights(), lr=lr,
                                             classes=(classes), drop_out=drop, kernel=kernel, num_layers=num_layers,
                                             gru_units=gru_units, flags=[False, True, False, False])
    final_CRNN_3 = CRNN_construction_final_n(window_size, initial_model=initial_CRNN.get_weights(), lr=lr,
                                             classes=(classes), drop_out=drop, kernel=kernel, num_layers=num_layers,
                                             gru_units=gru_units, flags=[False, False, True, False])
    final_CRNN_4 = CRNN_construction_final_n(window_size, initial_model=initial_CRNN.get_weights(), lr=lr,
                                             classes=(classes), drop_out=drop, kernel=kernel, num_layers=num_layers,
                                             gru_units=gru_units, flags=[False, False, False, True])
    final_CRNN_x.save_weights('best_current_.h5')

    MODEL = STU_TEACH_dyn(initial_CRNN, final_CRNN_x, final_CRNN_1, final_CRNN_2, final_CRNN_3, final_CRNN_4, classes)




optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
if classes == 2 and more == 'one':
                classes_monitored = classes + 1
else:
                classes_monitored = classes + 2
MODEL.compile(final_optimizer=optimizer,
                          loss={"KD_loss": network.metrics_losses.binary_crossentropy, "final_loss": network.metrics_losses.binary_crossentropy},
                          loss_weights=loss_weights, F1_score=StatefullF1(n_class=classes_monitored))

print('Train shape:',Y_train.shape)
print('Val shape:', Y_val.shape)
print('Test shape:', Y_test.shape)

history = MODEL.fit(x=X_train, y=Y_train, shuffle=True, epochs=1000, batch_size=batch_size_,
                                       validation_data=(X_test,Y_test), callbacks=[early_stop], verbose=1)

if approach == 'AIL':
    MODEL.final_x.save_weights('../CRNN_final_model_' + type_ + '.h5')
else:
    MODEL.final.save_weights('../CRNN_final_model_' + type_ + '.h5')

classes = classes_monitored

output_strong = MODEL.predict(x=X_val)
output_strong_test = MODEL.predict(x=X_test)


shape = output_strong.shape[0] * output_strong.shape[1]
shape_test = output_strong_test.shape[0] * output_strong_test.shape[1]

Y_val = Y_val.reshape(shape, classes)
Y_test = Y_test.reshape(shape_test, classes)


output_strong = output_strong.reshape(shape, classes)
output_strong_test = output_strong_test.reshape(shape_test, classes)



if thres_strong == 0:
                    thres_strong = thres_analysis(Y_val, output_strong,classes)

assert (Y_val.shape == output_strong.shape)
print(thres_strong)


output_strong_test = app_binarization_strong(output_strong_test, thres_strong, classes)
output_strong = app_binarization_strong(output_strong, thres_strong, classes)

print("Test")
print(multilabel_confusion_matrix(Y_test, output_strong_test))
print(classification_report(Y_test, output_strong_test))
