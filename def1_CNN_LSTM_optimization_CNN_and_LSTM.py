from inspect import EndOfBlock
import numpy as np
import matplotlib.pyplot as plt
import pandas as pandas
from keras.models import Sequential,Model
from keras.layers import *
import os
from scipy.io import loadmat
import time
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
import statistics
from scipy.stats import gaussian_kde
import tensorflow as tf
from scipy.stats import pearsonr
from tabulate import tabulate
import keras.backend as K 


# %% DATA
os.chdir(R'C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Matlab\FINAL_preparation_ML_models\CNN_400_Persons')
inputall = loadmat('CNN_lstm_esmaelpoor.mat')  
#inputall = loadmat('cnndata.mat')  

x_values=inputall['x_values']
y_values=inputall['ydsbp_values']
#y_values=inputall['ysdbp_values']

indices=range(len(x_values))
del inputall
x_train, x_test, y_train, y_test,indices_train,indices_test  =train_test_split(x_values, y_values,indices, test_size=0.20,random_state=1)
x_train, x_val, y_train, y_val,indices_train,indices_val =train_test_split(x_train, y_train, indices_train, test_size=0.25,random_state=1 )

x_train     = np.expand_dims(x_train, axis=2)
x_test      = np.expand_dims(x_test, axis=2)
x_val       =np.expand_dims(x_val, axis=2)

# x_train     = np.expand_dims(x_train, axis=3)
# x_test      = np.expand_dims(x_test, axis=3)
# x_val       =np.expand_dims(x_val, axis=3)

# y_train=np.expand_dims(y_train,axis=1)
# y_val=np.expand_dims(y_val,axis=1)
# y_test=np.expand_dims(y_test,axis=1)

# %% MODEL
Epochsdbpcnn = 0#100#50
Epochssbpcnn = 0#100#50
batch_size_dbp = 95
batch_size_sbp = 68
Filtersdbp= 47
kernelsdbp = 21
Filterssbp= 41
kernelssbp = 27
plotlosses=1

#%% CNN model DBP
modeldbp=Sequential(name='DBP_CNN_model')
modeldbp.add(Conv1D(filters=Filtersdbp, kernel_size=kernelsdbp,padding='same',activation='relu',input_shape=(x_train.shape[1],1)))
modeldbp.add(Conv1D(filters=Filtersdbp, kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(Conv1D(filters=Filtersdbp, kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(MaxPooling1D(2,padding='same'))
modeldbp.add(Conv1D(filters=Filtersdbp*2, kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(Conv1D(filters=Filtersdbp*2, kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(Conv1D(filters=Filtersdbp*2, kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(Conv1D(filters=Filtersdbp*2, kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(Conv1D(filters=Filtersdbp*2, kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(Conv1D(filters=Filtersdbp*2, kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(Conv1D(filters=Filtersdbp*2, kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(Conv1D(filters=Filtersdbp*2, kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(MaxPooling1D(2,padding='same'))
modeldbp.add(Conv1D(filters=Filtersdbp, kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(Conv1D(filters=Filtersdbp, kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(Conv1D(filters=Filtersdbp, kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(MaxPooling1D(2,padding='same'))
modeldbp.add(Conv1D(filters=int(Filtersdbp/2), kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(Conv1D(filters=int(Filtersdbp/2), kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(Conv1D(filters=int(Filtersdbp/2), kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(Conv1D(filters=int(Filtersdbp/2), kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(Conv1D(filters=int(Filtersdbp/2), kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(Conv1D(filters=int(Filtersdbp/2), kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(Conv1D(filters=int(Filtersdbp/2), kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(Conv1D(filters=int(Filtersdbp/2), kernel_size=kernelsdbp,padding='same',activation='relu'))
modeldbp.add(MaxPooling1D(2,padding='same'))
modeldbp.add(Flatten(name='FeaturevectorDBP'))
modeldbp.add(Dense(1,name='DBPest'))

#%% CNN model SBP
modelsbp = Sequential(name='SBP_CNN_model')
modelsbp.add(Conv1D(filters=Filterssbp, kernel_size=kernelssbp,padding='same',activation='relu',input_shape=(x_train.shape[1],1)))
modelsbp.add(Conv1D(filters=Filterssbp,kernel_size=kernelssbp,padding='same', activation='relu'))
modelsbp.add(Conv1D(filters=Filterssbp,kernel_size=kernelssbp,padding='same', activation='relu'))
modelsbp.add(MaxPooling1D(2,padding='same'))
modelsbp.add(Conv1D(filters=Filterssbp*2,kernel_size=kernelssbp,padding='same', activation='relu'))
modelsbp.add(Conv1D(filters=Filterssbp*2,kernel_size=kernelssbp,padding='same', activation='relu'))
modelsbp.add(Conv1D(filters=Filterssbp*2,kernel_size=kernelssbp,padding='same', activation='relu'))
modelsbp.add(MaxPooling1D(2,padding='same'))
modelsbp.add(Conv1D(filters=Filterssbp,kernel_size=kernelssbp,padding='same', activation='relu'))
modelsbp.add(MaxPooling1D(2,padding='same'))
modelsbp.add(Conv1D(filters=int(Filterssbp/2),kernel_size=kernelssbp,padding='same', activation='relu'))
modelsbp.add(Conv1D(filters=int(Filterssbp/2),kernel_size=kernelssbp,padding='same', activation='relu'))
modelsbp.add(Conv1D(filters=int(Filterssbp/2),kernel_size=kernelssbp,padding='same', activation='relu'))
modelsbp.add(MaxPooling1D(2,padding='same'))
modelsbp.add(Flatten(name='FeaturevectorSBP'))
modelsbp.add(Dense(1,name='SBPest'))

plot_model(modeldbp, to_file=R'C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\ML_results\Def1_CNN+LSTM\CNN+LSTMresults\CNNmodel_cnn_lstm_bayes_opt_dbp.png', show_shapes=True, show_layer_names=True)
plot_model(modelsbp, to_file=R'C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\ML_results\Def1_CNN+LSTM\CNN+LSTMresults\CNNmodel_cnn_lstm_bayes_opt_sbp.png', show_shapes=True, show_layer_names=True)

modeldbp.summary()
modelsbp.summary()

modeldbp.compile(optimizer='adam', loss='mean_absolute_error',metrics=['mae', 'mse'])
modelsbp.compile(optimizer='adam', loss='mean_absolute_error',metrics=['mae', 'mse'])

#checkpoint_path_dbpcnn = R"C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Results_LSTM_CNN_combined\Bayes_opt\featureweights_cnn_lstm_bayes_opt_dbpcnn.ckpt" #50epochs
checkpoint_path_dbpcnn = R"C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Results_LSTM_CNN_combined\Bayes_opt\100_epochs_withCNN_LSTM_Bayes_opt50epochs\fw_cnn_lstm_bayes_opt_dbpcnn_100epochs.ckpt" #100epochs
checkpoint_dir_dbpcnn = os.path.dirname(checkpoint_path_dbpcnn)

#checkpoint_path_sbpcnn = R"C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Results_LSTM_CNN_combined\Bayes_opt\featureweights_cnn_lstm_bayes_opt_sbpcnn.ckpt" #50epochs
checkpoint_path_sbpcnn = R"C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Results_LSTM_CNN_combined\Bayes_opt\100_epochs_withCNN_LSTM_Bayes_opt50epochs\fw_cnn_lstm_bayes_opt_sbpcnn_100epochs.ckpt" #100epochs
checkpoint_dir_sbpcnn = os.path.dirname(checkpoint_path_sbpcnn)
# Loads the weights
modeldbp.load_weights(checkpoint_path_dbpcnn)
modelsbp.load_weights(checkpoint_path_sbpcnn)

# Create a callback that saves the model's weights
cp_callback_dbpcnn = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_dbpcnn,
                                                 save_weights_only=True,
                                                 verbose=1)

cp_callback_sbpcnn = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_sbpcnn,
                                                 save_weights_only=True,
                                                 verbose=1)
#train model DBP
start_time = time.time()
history_dbpcnn=modeldbp.fit(x_train, y_train[:,0], batch_size=batch_size_dbp, 
                    epochs=Epochsdbpcnn,validation_data=(x_val,y_val[:,0]), verbose=1,callbacks=[cp_callback_dbpcnn])#, callbacks=[es])
print("--- %s seconds ---" % (time.time() - start_time))

#train model SBP
start_time = time.time()
history_sbpcnn=modelsbp.fit(x_train, y_train[:,1], batch_size=batch_size_sbp, 
                    epochs=Epochssbpcnn,validation_data=(x_val,y_val[:,1]), verbose=1,callbacks=[cp_callback_sbpcnn])#, callbacks=[es])
print("--- %s seconds ---" % (time.time() - start_time))

# Prediction CNN
val_predictions_dbpcnn = modeldbp.predict(x_val)
val_predictions_sbpcnn = modelsbp.predict(x_val)

difference_dbp_cnn=y_val[:,0]-val_predictions_dbpcnn[:,0]
meanerror_dbp_cnn=statistics.mean(difference_dbp_cnn)
stdev_dbp_cnn=statistics.stdev(difference_dbp_cnn)
difference_sbp_cnn=y_val[:,1]-val_predictions_sbpcnn[:,0]
meanerror_sbp_cnn=statistics.mean(difference_sbp_cnn)
stdev_sbp_cnn=statistics.stdev(difference_sbp_cnn)

print('DBP CNN: ',meanerror_dbp_cnn,stdev_dbp_cnn)
print('SBP CNN: ',meanerror_sbp_cnn,stdev_sbp_cnn)

# Feature vectors DBP
layer_name = 'FeaturevectorDBP'
intermediate_layer_model = Model(inputs=modeldbp.input,
                                 outputs=modeldbp.get_layer(layer_name).output)
feature_vector_dbp_train = intermediate_layer_model.predict(x_train)
feature_vector_dbp_val = intermediate_layer_model.predict(x_val)
feature_vector_dbp_test = intermediate_layer_model.predict(x_test)

layer_name = 'DBPest'
output_layer_model = Model(inputs=modeldbp.input,
                                 outputs=modeldbp.get_layer(layer_name).output)
feature_vector_dbp_train_est = output_layer_model.predict(x_train)
feature_vector_dbp_val_est = output_layer_model.predict(x_val)
feature_vector_dbp_test_est = output_layer_model.predict(x_test)

#Feature vectors SBP
layer_name = 'FeaturevectorSBP'
intermediate_layer_model = Model(inputs=modelsbp.input,
                                 outputs=modelsbp.get_layer(layer_name).output)
feature_vector_sbp_train = intermediate_layer_model.predict(x_train)
feature_vector_sbp_val = intermediate_layer_model.predict(x_val)
feature_vector_sbp_test = intermediate_layer_model.predict(x_test)

layer_name = 'SBPest'
output_layer_model = Model(inputs=modelsbp.input,
                                 outputs=modelsbp.get_layer(layer_name).output)
feature_vector_sbp_train_est = output_layer_model.predict(x_train)
feature_vector_sbp_val_est = output_layer_model.predict(x_val)
feature_vector_sbp_test_est = output_layer_model.predict(x_test)

#Input vectors
input_lstm_dbp_train = np.concatenate((feature_vector_dbp_train,feature_vector_sbp_train_est), axis = 1)
input_lstm_dbp_val = np.concatenate((feature_vector_dbp_val,feature_vector_sbp_val_est), axis = 1)
input_lstm_dbp_test = np.concatenate((feature_vector_dbp_test,feature_vector_sbp_test_est), axis = 1)
input_lstm_sbp_train = np.concatenate((feature_vector_sbp_train,feature_vector_dbp_train_est), axis = 1)
input_lstm_sbp_val = np.concatenate((feature_vector_sbp_val,feature_vector_dbp_val_est), axis = 1)
input_lstm_sbp_test = np.concatenate((feature_vector_sbp_test,feature_vector_dbp_test_est), axis = 1)

input_lstm_dbp_train     = np.expand_dims(input_lstm_dbp_train, axis=2)
input_lstm_dbp_val     = np.expand_dims(input_lstm_dbp_val, axis=2)
input_lstm_dbp_test     = np.expand_dims(input_lstm_dbp_test, axis=2)
input_lstm_sbp_train     = np.expand_dims(input_lstm_sbp_train, axis=2)
input_lstm_sbp_val     = np.expand_dims(input_lstm_sbp_val, axis=2)
input_lstm_sbp_test     = np.expand_dims(input_lstm_sbp_test, axis=2)

#%% LSTM model DBP
Epochsdbplstm = 0#100 #50
Epochssbplstm = 0#100 #50
batchsize_dbp_lstm=67#32
batchsize_sbp_lstm=24#99
dropper_dbp=0.4055#0.09443
dropper_sbp=0.5421#0.4938

#LSTM Part DBP
modeldbplstm=Sequential()
modeldbplstm.add(CuDNNLSTM(122,return_sequences=True, input_shape=(input_lstm_dbp_train.shape[1],1))) #71
modeldbplstm.add(Dropout(dropper_dbp))
modeldbplstm.add(CuDNNLSTM(92, return_sequences=True)) #27
modeldbplstm.add(CuDNNLSTM(92, return_sequences=True)) #27
modeldbplstm.add(Dropout(dropper_dbp))
modeldbplstm.add(CuDNNLSTM(82, return_sequences=False)) #132
modeldbplstm.add(Dense(239)) #120
modeldbplstm.add(Dense(1))

#LSTM Part SBP
modelsbplstm=Sequential()
modelsbplstm.add(CuDNNLSTM(123,return_sequences=True, input_shape=(input_lstm_sbp_train.shape[1],1))) #216
modelsbplstm.add(Dropout(dropper_sbp))
modelsbplstm.add(CuDNNLSTM(52, return_sequences=True)) #54, was maar 1 laag hier bij 1ste bayes opt met 50 epochs
modelsbplstm.add(CuDNNLSTM(52, return_sequences=True))
modelsbplstm.add(Dropout(dropper_sbp))
modelsbplstm.add(CuDNNLSTM(12, return_sequences=False)) #160
modelsbplstm.add(Dense(186)) #113
modelsbplstm.add(Dense(1))

plot_model(modeldbplstm, to_file=R'C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\ML_results\Def1_CNN+LSTM\CNN+LSTMresults\LSTMmodel_cnn_lstm_bayesopt_dbplstm.png', show_shapes=True, show_layer_names=True)
plot_model(modelsbplstm, to_file=R'C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\ML_results\Def1_CNN+LSTM\CNN+LSTMresults\LSTMmodel_cnn_lstm_bayesopt_sbplstm.png', show_shapes=True, show_layer_names=True)

modeldbplstm.summary()
modelsbplstm.summary()

modeldbplstm.compile(optimizer='adam', loss='mean_squared_error',metrics=['mae', 'mse'])
modelsbplstm.compile(optimizer='adam', loss='mean_squared_error',metrics=['mae', 'mse'])

#checkpoint_path_dbplstm = R"C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Results_LSTM_CNN_combined\Esmaelpoor\featureweights_cnn_lstm_esmaelpoor_dbplstm_2lstm2dense.ckpt"
#checkpoint_path_dbplstm = R"C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Results_LSTM_CNN_combined\Esmaelpoorarchitecture\Analysis\featureweights_cnn_lstm_esmaelpoor_dbplstm_2lstm2densetest.ckpt" #bayes_lstm_50epochs
#checkpoint_path_dbplstm = R"C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Results_LSTM_CNN_combined\Bayes_opt\100_epochs_withCNN_LSTM_Bayes_opt50epochs\fw_cnn_lstm_bayes_opt_dbplstm_50epochs.ckpt" #50epochs with first bayesianlstm
checkpoint_path_dbplstm = R"C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Results_LSTM_CNN_combined\Bayes_opt_100epochs\fw_cnnlstm_bayes_opt_dbplstm_100epochs.ckpt" #100epochs aftersecond time bayes opt
#checkpoint_path_dbplstm = R"C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Results_LSTM_CNN_combined\Bayes_opt_100epochs\fw_cnnlstm_bayes_opt_dbplstm_2nd_100epochs.ckpt" #2nd 100epochs:total of 200epochs aftersecond time bayes opt
checkpoint_dir_dbplstm = os.path.dirname(checkpoint_path_dbplstm)




#checkpoint_path_sbplstm = R"C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Results_LSTM_CNN_combined\Esmaelpoor\featureweights_cnn_lstm_esmaelpoor_sbplstm_2lstm2dense.ckpt"
#checkpoint_path_sbplstm = R"C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Results_LSTM_CNN_combined\Esmaelpoorarchitecture\Analysis\featureweights_cnn_lstm_esmaelpoor_sbplstm_2lstm2densetest.ckpt"#bayes_lstm_50epochs
#checkpoint_path_sbplstm = R"C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Results_LSTM_CNN_combined\Bayes_opt\100_epochs_withCNN_LSTM_Bayes_opt50epochs\fw_cnn_lstm_bayes_opt_sbplstm_50epochs.ckpt"#50epochs with first bayesianlstm
checkpoint_path_sbplstm = R"C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Results_LSTM_CNN_combined\Bayes_opt_100epochs\fw_cnn_lstm_bayes_opt_sbplstm_100epochs.ckpt"#100epochs aftersecond time bayes opt
#checkpoint_path_sbplstm = R"C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Results_LSTM_CNN_combined\Bayes_opt_100epochs\fw_cnnlstm_bayes_opt_sbplstm_2nd_100epochs.ckpt"#2nd 100epochs:total of 200epochs aftersecond time bayes opt
checkpoint_dir_sbplstm = os.path.dirname(checkpoint_path_sbplstm)

# Loads the weights
modeldbplstm.load_weights(checkpoint_path_dbplstm)
modelsbplstm.load_weights(checkpoint_path_sbplstm)


# Create a callback that saves the model's weights
cp_callback_dbplstm = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_dbplstm,
                                                 save_weights_only=True,
                                                 verbose=1)

cp_callback_sbplstm = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_sbplstm,
                                                 save_weights_only=True,
                                                 verbose=1)
#train model DBP
start_time = time.time()
history_dbplstm=modeldbplstm.fit(input_lstm_dbp_train, y_train[:,0], batch_size=batchsize_dbp_lstm, 
                    epochs=Epochsdbplstm,validation_data=(input_lstm_dbp_val,y_val[:,0]), verbose=1,callbacks=[cp_callback_dbplstm])#, callbacks=[es])
print("--- %s seconds ---" % (time.time() - start_time))

#train model SBP
start_time = time.time()
history_sbplstm=modelsbplstm.fit(input_lstm_sbp_train, y_train[:,1], batch_size=batchsize_sbp_lstm, 
                    epochs=Epochssbplstm,validation_data=(input_lstm_sbp_val,y_val[:,1]), verbose=1,callbacks=[cp_callback_sbplstm])#, callbacks=[es])
print("--- %s seconds ---" % (time.time() - start_time))

#%% predictions
val_predictions_dbplstm = modeldbplstm.predict(input_lstm_dbp_val)
val_predictions_sbplstm = modelsbplstm.predict(input_lstm_sbp_val)

difference_dbp_cnn=y_val[:,0]-val_predictions_dbpcnn[:,0]
meanerror_dbp_cnn=statistics.mean(difference_dbp_cnn)
stdev_dbp_cnn=statistics.stdev(difference_dbp_cnn)

difference_dbp_lstm=y_val[:,0]-val_predictions_dbplstm[:,0]
meanerror_dbp_lstm=statistics.mean(difference_dbp_lstm)
stdev_dbp_lstm=statistics.stdev(difference_dbp_lstm)

difference_sbp_cnn=y_val[:,1]-val_predictions_sbpcnn[:,0]
meanerror_sbp_cnn=statistics.mean(difference_sbp_cnn)
stdev_sbp_cnn=statistics.stdev(difference_sbp_cnn)

difference_sbp_lstm=y_val[:,1]-val_predictions_sbplstm[:,0]
meanerror_sbp_lstm=statistics.mean(difference_sbp_lstm)
stdev_sbp_lstm=statistics.stdev(difference_sbp_lstm)

print('DBP CNN: ',meanerror_dbp_cnn,stdev_dbp_cnn)
print('SBP CNN: ',meanerror_sbp_cnn,stdev_sbp_cnn)
print('DBP LSTM: ',meanerror_dbp_lstm,stdev_dbp_lstm)
print('SBP LSTM: ',meanerror_sbp_lstm,stdev_sbp_lstm)
if Epochsdbpcnn>0:
    # Losses
    mse_train_dbpcnn   = history_dbpcnn.history['mse']
    mse_val_dbpcnn     = history_dbpcnn.history['val_mse']
    mae_train_dbpcnn   = history_dbpcnn.history['mae']
    mae_val_dbpcnn     = history_dbpcnn.history['val_mae']

    mse_train_sbpcnn   = history_sbpcnn.history['mse']
    mse_val_sbpcnn     = history_sbpcnn.history['val_mse']
    mae_train_sbpcnn   = history_sbpcnn.history['mae']
    mae_val_sbpcnn     = history_sbpcnn.history['val_mae']

        #Loss plots
    plt.figure(figsize=(9, 3))
    plt.subplot(121)
    plt.plot(mae_train_dbpcnn)
    plt.plot(mae_val_dbpcnn)
    plt.title('Model MAE: Diastolic Blood Pressure CNN')
    #plt.ylim([0,25])
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.yscale('log')
    plt.legend(['Train', 'Val'], loc='upper left')
    # summarize history for loss
    plt.subplot(122)
    plt.plot(mse_train_dbpcnn)
    plt.plot(mse_val_dbpcnn)
    plt.title('Model MSE: Diastolic Blood Pressure CNN')
    #plt.ylim([0,500])
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epochs')
    plt.yscale('log')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    plt.figure(figsize=(9, 3))
    plt.subplot(121)
    plt.plot(mae_train_sbpcnn)
    plt.plot(mae_val_sbpcnn)
    plt.title('Model MAE: Systolic Blood Pressure CNN')
    #plt.ylim([0,25])
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.yscale('log')
    plt.legend(['Train', 'Val'], loc='upper left')
    # summarize history for loss
    plt.subplot(122)
    plt.plot(mse_train_sbpcnn)
    plt.plot(mse_val_sbpcnn)
    plt.title('Model MSE: Systolic Blood Pressure CNN')
    #plt.ylim([0,500])
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epochs')
    plt.yscale('log')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


if Epochsdbplstm>0:
    mse_train_dbplstm   = history_dbplstm.history['mse']
    mse_val_dbplstm     = history_dbplstm.history['val_mse']
    mae_train_dbplstm   = history_dbplstm.history['mae']
    mae_val_dbplstm     = history_dbplstm.history['val_mae']

    mse_train_sbplstm   = history_sbplstm.history['mse']
    mse_val_sbplstm     = history_sbplstm.history['val_mse']
    mae_train_sbplstm   = history_sbplstm.history['mae']
    mae_val_sbplstm     = history_sbplstm.history['val_mae']

    # Loss plots LSTM
    plt.figure(figsize=(9, 3))
    plt.subplot(121)
    plt.plot(mae_train_dbplstm)
    plt.plot(mae_val_dbplstm)
    plt.title('Model MAE: Diastolic Blood Pressure LSTM')
    #plt.ylim([0,25])
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.yscale('log')
    plt.legend(['Train', 'Val'], loc='upper left')
    # summarize history for loss
    plt.subplot(122)
    plt.plot(mse_train_dbplstm)
    plt.plot(mse_val_dbplstm)
    plt.title('Model MSE: Diastolic Blood Pressure LSTM')
    #plt.ylim([0,500])
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epochs')
    plt.yscale('log')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    plt.figure(figsize=(9, 3))
    plt.subplot(121)
    plt.plot(mae_train_sbplstm)
    plt.plot(mae_val_sbplstm)
    plt.title('Model MAE: Systolic Blood Pressure LSTM')
    #plt.ylim([0,25])
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.yscale('log')
    plt.legend(['Train', 'Val'], loc='upper left')
    # summarize history for loss
    plt.subplot(122)
    plt.plot(mse_train_sbplstm)
    plt.plot(mse_val_sbplstm)
    plt.title('Model MSE: Systolic Blood Pressure LSTM')
    #plt.ylim([0,500])
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epochs')
    plt.yscale('log')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


[Rvaluedbp_cnn,pvaluedbp_cnn]= pearsonr(y_val[:,0],val_predictions_dbpcnn[:,0])

[Rvaluesbp_cnn,pvaluesbp_cnn]= pearsonr(y_val[:,1],val_predictions_sbpcnn[:,0])

totalamount=y_val.shape[0]
# BHS standards DBP
differencesdbp_cnn=np.abs(difference_dbp_cnn)
listlower5dbp_cnn = [i for i in differencesdbp_cnn if i < 5]
listlower10dbp_cnn = [i for i in differencesdbp_cnn if i < 10]
listlower15dbp_cnn = [i for i in differencesdbp_cnn if i < 15]
percentage5dbp_cnn=(len(listlower5dbp_cnn)/totalamount)*100
percentage10dbp_cnn=(len(listlower10dbp_cnn)/totalamount)*100
percentage15dbp_cnn=(len(listlower15dbp_cnn)/totalamount)*100

# BHS standards SBP
differencessbp_cnn=np.abs(difference_sbp_cnn)
listlower5sbp_cnn = [i for i in differencessbp_cnn if i < 5]
listlower10sbp_cnn = [i for i in differencessbp_cnn if i < 10]
listlower15sbp_cnn = [i for i in differencessbp_cnn if i < 15]
percentage5sbp_cnn=(len(listlower5sbp_cnn)/totalamount)*100
percentage10sbp_cnn=(len(listlower10sbp_cnn)/totalamount)*100
percentage15sbp_cnn=(len(listlower15sbp_cnn)/totalamount)*100

yvalpredictionsdbp = np.vstack([y_val[:,0],val_predictions_dbpcnn[:,0]])
zdbp = gaussian_kde(yvalpredictionsdbp)(yvalpredictionsdbp)

plt.figure()
plt.scatter(y_val[:,0], val_predictions_dbpcnn[:,0],c=zdbp,s=10) 
plt.plot([40, 140], [40, 140], '--k') 
plt.title('Correlation plot DBP validation set')
#plt.ylim([0,500])
plt.xlabel('Expected DBP')
plt.ylabel('Predictions DBP')

yvalpredictionssbp = np.vstack([y_val[:,1],val_predictions_sbpcnn[:,0]])
zsbp = gaussian_kde(yvalpredictionssbp)(yvalpredictionssbp)
plt.figure()
plt.scatter(y_val[:,1], val_predictions_sbpcnn[:,0],c=zsbp,s=10) 
plt.plot([80, 200], [80, 200], '--k') 
plt.title('Correlation plot SBP validation set')
#plt.ylim([0,500])
plt.xlabel('Expected SBP')
plt.ylabel('Predictions SBP')

#BHS table
table_val_cnn=[['CNN','<5 mmHg error [%]', '<10 mmHg error [%]', '<15 mmHg error [%]', 'BHS grade','R', 'Mean error','Standard dev'],['DBP',percentage5dbp_cnn,percentage10dbp_cnn,percentage15dbp_cnn,' ' ,Rvaluedbp_cnn,meanerror_dbp_cnn,stdev_dbp_cnn],['SBP',percentage5sbp_cnn,percentage10sbp_cnn,percentage15sbp_cnn, ' ',Rvaluesbp_cnn,meanerror_sbp_cnn,stdev_sbp_cnn ]]
print(tabulate(table_val_cnn, headers='firstrow', tablefmt='fancy_grid'))

[Rvaluedbp_lstm,pvaluedbp_lstm]= pearsonr(y_val[:,0],val_predictions_dbplstm[:,0])

[Rvaluesbp_lstm,pvaluesbp_lstm]= pearsonr(y_val[:,1],val_predictions_sbplstm[:,0])

totalamount=y_val.shape[0]
# BHS standards DBP
differencesdbp_lstm=np.abs(difference_dbp_lstm)
listlower5dbp_lstm = [i for i in differencesdbp_lstm if i < 5]
listlower10dbp_lstm = [i for i in differencesdbp_lstm if i < 10]
listlower15dbp_lstm = [i for i in differencesdbp_lstm if i < 15]
percentage5dbp_lstm=(len(listlower5dbp_lstm)/totalamount)*100
percentage10dbp_lstm=(len(listlower10dbp_lstm)/totalamount)*100
percentage15dbp_lstm=(len(listlower15dbp_lstm)/totalamount)*100

# BHS standards SBP
differencessbp_lstm=np.abs(difference_sbp_lstm)
listlower5sbp_lstm = [i for i in differencessbp_lstm if i < 5]
listlower10sbp_lstm = [i for i in differencessbp_lstm if i < 10]
listlower15sbp_lstm = [i for i in differencessbp_lstm if i < 15]
percentage5sbp_lstm=(len(listlower5sbp_lstm)/totalamount)*100
percentage10sbp_lstm=(len(listlower10sbp_lstm)/totalamount)*100
percentage15sbp_lstm=(len(listlower15sbp_lstm)/totalamount)*100

yvalpredictionsdbplstm = np.vstack([y_val[:,0],val_predictions_dbplstm[:,0]])
zdbplstm = gaussian_kde(yvalpredictionsdbplstm)(yvalpredictionsdbplstm)

plt.figure()
plt.scatter(y_val[:,0], val_predictions_dbplstm[:,0],c=zdbplstm,s=10) 
plt.plot([40, 140], [40, 140], '--k') 
plt.title('Correlation plot DBP validation set')
#plt.ylim([0,500])
plt.xlabel('Expected DBP')
plt.ylabel('Predictions DBP')

yvalpredictionssbplstm = np.vstack([y_val[:,1],val_predictions_sbplstm[:,0]])
zsbplstm = gaussian_kde(yvalpredictionssbplstm)(yvalpredictionssbplstm)
plt.figure()
plt.scatter(y_val[:,1], val_predictions_sbplstm[:,0],c=zsbplstm,s=10) 
plt.plot([80, 200], [80, 200], '--k') 
plt.title('Correlation plot SBP validation set')
#plt.ylim([0,500])
plt.xlabel('Expected SBP')
plt.ylabel('Predictions SBP')

#BHS table
table_val_lstm=[['CNN+LSTM','<5 mmHg error [%]', '<10 mmHg error [%]', '<15 mmHg error [%]', 'BHS grade','R', 'Mean error','Standard dev'],['DBP',percentage5dbp_lstm,percentage10dbp_lstm,percentage15dbp_lstm,' ' ,Rvaluedbp_lstm,meanerror_dbp_lstm,stdev_dbp_lstm],['SBP',percentage5sbp_lstm,percentage10sbp_lstm,percentage15sbp_lstm, ' ',Rvaluesbp_lstm,meanerror_sbp_lstm,stdev_sbp_lstm ]]
print(tabulate(table_val_lstm, headers='firstrow', tablefmt='fancy_grid'))

#%% Test predictions
test_predictions_dbpcnn = modeldbp.predict(x_test)
test_predictions_sbpcnn = modelsbp.predict(x_test)
test_predictions_dbplstm = modeldbplstm.predict(input_lstm_dbp_test)
test_predictions_sbplstm = modelsbplstm.predict(input_lstm_sbp_test)

difference_dbp_cnn_test=y_test[:,0]-test_predictions_dbpcnn[:,0]
meanerror_dbp_cnn_test=statistics.mean(difference_dbp_cnn_test)
stdev_dbp_cnn_test=statistics.stdev(difference_dbp_cnn_test)

difference_dbp_lstm_test=y_test[:,0]-test_predictions_dbplstm[:,0]
meanerror_dbp_lstm_test=statistics.mean(difference_dbp_lstm_test)
stdev_dbp_lstm_test=statistics.stdev(difference_dbp_lstm_test)

difference_sbp_cnn_test=y_test[:,1]-test_predictions_sbpcnn[:,0]
meanerror_sbp_cnn_test=statistics.mean(difference_sbp_cnn_test)
stdev_sbp_cnn_test=statistics.stdev(difference_sbp_cnn_test)

difference_sbp_lstm_test=y_test[:,1]-test_predictions_sbplstm[:,0]
meanerror_sbp_lstm_test=statistics.mean(difference_sbp_lstm_test)
stdev_sbp_lstm_test=statistics.stdev(difference_sbp_lstm_test)

print('DBP CNN: ',meanerror_dbp_cnn_test,stdev_dbp_cnn_test)
print('SBP CNN: ',meanerror_sbp_cnn_test,stdev_sbp_cnn_test)
print('DBP LSTM: ',meanerror_dbp_lstm_test,stdev_dbp_lstm_test)
print('SBP LSTM: ',meanerror_sbp_lstm_test,stdev_sbp_lstm_test)



[Rvaluedbp_cnn_test,pvaluedbp_cnn_test]= pearsonr(y_test[:,0],test_predictions_dbpcnn[:,0])

[Rvaluesbp_cnn_test,pvaluesbp_cnn_test]= pearsonr(y_test[:,1],test_predictions_sbpcnn[:,0])

totalamount_test=y_test.shape[0]
# BHS standards DBP
differencesdbp_cnn_test=np.abs(difference_dbp_cnn_test)
listlower5dbp_cnn_test = [i for i in differencesdbp_cnn_test if i < 5]
listlower10dbp_cnn_test = [i for i in differencesdbp_cnn_test if i < 10]
listlower15dbp_cnn_test = [i for i in differencesdbp_cnn_test if i < 15]
percentage5dbp_cnn_test=(len(listlower5dbp_cnn_test)/totalamount_test)*100
percentage10dbp_cnn_test=(len(listlower10dbp_cnn_test)/totalamount_test)*100
percentage15dbp_cnn_test=(len(listlower15dbp_cnn_test)/totalamount_test)*100

# BHS standards SBP
differencessbp_cnn_test=np.abs(difference_sbp_cnn_test)
listlower5sbp_cnn_test = [i for i in differencessbp_cnn_test if i < 5]
listlower10sbp_cnn_test = [i for i in differencessbp_cnn_test if i < 10]
listlower15sbp_cnn_test = [i for i in differencessbp_cnn_test if i < 15]
percentage5sbp_cnn_test=(len(listlower5sbp_cnn_test)/totalamount_test)*100
percentage10sbp_cnn_test=(len(listlower10sbp_cnn_test)/totalamount_test)*100
percentage15sbp_cnn_test=(len(listlower15sbp_cnn_test)/totalamount_test)*100

ytestpredictionsdbp = np.vstack([y_test[:,0],test_predictions_dbpcnn[:,0]])
zdbp_test = gaussian_kde(ytestpredictionsdbp)(ytestpredictionsdbp)

plt.figure()
plt.scatter(y_test[:,0], test_predictions_dbpcnn[:,0],c=zdbp_test,s=10) 
plt.plot([40, 140], [40, 140], '--k') 
plt.title('Correlation plot Diastolic Blood Pressure ')
plt.ylabel('Predicted DBP [mmHg]')
plt.xlabel('Actual DBP [mmHg]')

ytestpredictionssbp = np.vstack([y_test[:,1],test_predictions_sbpcnn[:,0]])
zsbp_test = gaussian_kde(ytestpredictionssbp)(ytestpredictionssbp)
plt.figure()
plt.scatter(y_test[:,1], test_predictions_sbpcnn[:,0],c=zsbp_test,s=10) 
plt.plot([80, 200], [80, 200], '--k') 
plt.title('Correlation plot Systolic Blood Pressure')
plt.ylabel('Predicted SBP [mmHg]')
plt.xlabel('Actual SBP [mmHg]')


#Bland altman plot diastolic blood pressure
data1     = np.asarray(y_test[:,0])
data2     = np.asarray(test_predictions_dbpcnn[:,0])
mean      = np.mean([data1, data2], axis=0)
diff      = data1 - data2                   # Difference between data1 and data2
md        = np.mean(diff)                   # Mean of the difference
sd        = np.std(diff, axis=0)            # Standard deviation of the difference

plt.figure()
plt.scatter(mean, diff, c=zdbp_test,s=10)
plt.axhline(md,           color='gray', linestyle='--')
plt.text(125,md+1.96*sd+0.4, '+sd1.96', fontsize=10)
plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
plt.text(125,md+0.4, 'Mean diff', fontsize=10)
plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
plt.text(125,md-1.96*sd+0.4, '-sd1.96', fontsize=10)
plt.title('Bland-Altman Plot Diastolic Blood Pressure')
plt.xlabel('Diastolic Blood Pressure [mmHg]')
plt.ylabel('Error [mmHg]')
plt.show()


#Bland altman plot Systolic blood pressure
data1     = np.asarray(y_test[:,1])
data2     = np.asarray(test_predictions_sbpcnn[:,0])
mean      = np.mean([data1, data2], axis=0)
diff      = data1 - data2                   # Difference between data1 and data2
md        = np.mean(diff)                   # Mean of the difference
sd        = np.std(diff, axis=0)            # Standard deviation of the difference

plt.figure()
plt.scatter(mean, diff, c=zsbp_test,s=10)
plt.axhline(md,           color='gray', linestyle='--')
plt.text(197,md+1.96*sd+0.4, '+sd1.96', fontsize=10)
plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
plt.text(197,md+0.4, 'Mean diff', fontsize=10)
plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
plt.text(197,md-1.96*sd+0.4, '-sd1.96', fontsize=10)
plt.title('Bland-Altman Plot Systolic Blood Pressure')
plt.xlabel('Systolic Blood Pressure [mmHg]')
plt.ylabel('Error [mmHg]')
plt.show()

#BHS table
table_test_cnn=[['CNN','<5 mmHg error [%]', '<10 mmHg error [%]', '<15 mmHg error [%]', 'BHS grade','R', 'Mean error','Standard dev'],['DBP',percentage5dbp_cnn_test,percentage10dbp_cnn_test,percentage15dbp_cnn_test,' ' ,Rvaluedbp_cnn_test,meanerror_dbp_cnn_test,stdev_dbp_cnn_test],['SBP',percentage5sbp_cnn_test,percentage10sbp_cnn_test,percentage15sbp_cnn_test, ' ',Rvaluesbp_cnn_test,meanerror_sbp_cnn_test,stdev_sbp_cnn_test ]]
print(tabulate(table_test_cnn, headers='firstrow', tablefmt='fancy_grid'))

[Rvaluedbp_lstm_test,pvaluedbp_lstm_test]= pearsonr(y_test[:,0],test_predictions_dbplstm[:,0])

[Rvaluesbp_lstm_test,pvaluesbp_lstm_test]= pearsonr(y_test[:,1],test_predictions_sbplstm[:,0])

totalamount_test=y_test.shape[0]
# BHS standards DBP
differencesdbp_lstm_test=np.abs(difference_dbp_lstm_test)
listlower5dbp_lstm_test = [i for i in differencesdbp_lstm_test if i < 5]
listlower10dbp_lstm_test = [i for i in differencesdbp_lstm_test if i < 10]
listlower15dbp_lstm_test = [i for i in differencesdbp_lstm_test if i < 15]
percentage5dbp_lstm_test=(len(listlower5dbp_lstm_test)/totalamount_test)*100
percentage10dbp_lstm_test=(len(listlower10dbp_lstm_test)/totalamount_test)*100
percentage15dbp_lstm_test=(len(listlower15dbp_lstm_test)/totalamount_test)*100

# BHS standards SBP
differencessbp_lstm_test=np.abs(difference_sbp_lstm_test)
listlower5sbp_lstm_test = [i for i in differencessbp_lstm_test if i < 5]
listlower10sbp_lstm_test = [i for i in differencessbp_lstm_test if i < 10]
listlower15sbp_lstm_test = [i for i in differencessbp_lstm_test if i < 15]
percentage5sbp_lstm_test=(len(listlower5sbp_lstm_test)/totalamount_test)*100
percentage10sbp_lstm_test=(len(listlower10sbp_lstm_test)/totalamount_test)*100
percentage15sbp_lstm_test=(len(listlower15sbp_lstm_test)/totalamount_test)*100

ytestpredictionsdbplstm = np.vstack([y_test[:,0],test_predictions_dbplstm[:,0]])
zdbplstm_test = gaussian_kde(ytestpredictionsdbplstm)(ytestpredictionsdbplstm)

plt.figure()
plt.scatter(y_test[:,0], test_predictions_dbplstm[:,0],c=zdbplstm_test,s=10) 
plt.plot([40, 140], [40, 140], '--k') 
plt.title('Correlation plot Diastolic Blood Pressure ')
plt.ylabel('Predicted DBP [mmHg]')
plt.xlabel('Actual DBP [mmHg]')

ytestpredictionssbplstm = np.vstack([y_test[:,1],test_predictions_sbplstm[:,0]])
zsbplstm_test = gaussian_kde(ytestpredictionssbplstm)(ytestpredictionssbplstm)
plt.figure()
plt.scatter(y_test[:,1], test_predictions_sbplstm[:,0],c=zsbplstm_test,s=10) 
plt.plot([80, 200], [80, 200], '--k') 
plt.title('Correlation plot Systolic Blood Pressure')
plt.ylabel('Predicted SBP [mmHg]')
plt.xlabel('Actual SBP [mmHg]')

#Bland altman plot diastolic blood pressure
data1     = np.asarray(y_test[:,0])
data2     = np.asarray(test_predictions_dbplstm[:,0])
mean      = np.mean([data1, data2], axis=0)
diff      = data1 - data2                   # Difference between data1 and data2
md        = np.mean(diff)                   # Mean of the difference
sd        = np.std(diff, axis=0)            # Standard deviation of the difference

plt.figure()
plt.scatter(mean, diff, c=zdbplstm_test,s=10)
plt.axhline(md,           color='gray', linestyle='--')
plt.text(122,md+1.96*sd+0.4, '+sd1.96', fontsize=10)
plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
plt.text(122,md+0.4, 'Mean diff', fontsize=10)
plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
plt.text(122,md-1.96*sd+0.4, '-sd1.96', fontsize=10)
plt.title('Bland-Altman Plot Diastolic Blood Pressure')
plt.xlabel('Diastolic Blood Pressure [mmHg]')
plt.ylabel('Error [mmHg]')
plt.show()


#Bland altman plot Systolic blood pressure
data1     = np.asarray(y_test[:,1])
data2     = np.asarray(test_predictions_sbplstm[:,0])
mean      = np.mean([data1, data2], axis=0)
diff      = data1 - data2                   # Difference between data1 and data2
md        = np.mean(diff)                   # Mean of the difference
sd        = np.std(diff, axis=0)            # Standard deviation of the difference

plt.figure()
plt.scatter(mean, diff, c=zsbplstm_test,s=10)
plt.axhline(md,           color='gray', linestyle='--')
plt.text(84,md+1.96*sd+0.4, '+sd1.96', fontsize=10)
plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
plt.text(84,md+0.4, 'Mean diff', fontsize=10)
plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
plt.text(84,md-1.96*sd+0.4, '-sd1.96', fontsize=10)
plt.title('Bland-Altman Plot Systolic Blood Pressure')
plt.xlabel('Systolic Blood Pressure [mmHg]')
plt.ylabel('Error [mmHg]')
plt.show()

#BHS table
table_test_lstm=[['CNN+LSTM','<5 mmHg error [%]', '<10 mmHg error [%]', '<15 mmHg error [%]', 'BHS grade','R', 'Mean error','Standard dev'],['DBP',percentage5dbp_lstm_test,percentage10dbp_lstm_test,percentage15dbp_lstm_test,' ' ,Rvaluedbp_lstm_test,meanerror_dbp_lstm_test,stdev_dbp_lstm_test],['SBP',percentage5sbp_lstm_test,percentage10sbp_lstm_test,percentage15sbp_lstm_test, ' ',Rvaluesbp_lstm_test,meanerror_sbp_lstm_test,stdev_sbp_lstm_test ]]
print(tabulate(table_test_lstm, headers='firstrow', tablefmt='fancy_grid'))