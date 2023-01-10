from keras.utils.vis_utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pandas
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, TimeDistributed, CuDNNLSTM
import os
from scipy.io import loadmat
import time
from sklearn.model_selection import train_test_split
import statistics
from scipy.stats import gaussian_kde
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from scipy.stats import pearsonr
from tabulate import tabulate
from keras.optimizers import adam_v2
from sklearn.metrics import mean_absolute_error,mean_squared_error
# %% DATA
os.chdir(R'C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Matlab\FINAL_preparation_ML_models\CNN_400_Persons')
inputall = loadmat('cnndata.mat')  

x_values=inputall['x_values']
y_values=inputall['y_values']
indices=range(len(x_values))
del inputall

x_train, x_test, y_train, y_test,indices_train,indices_test  =train_test_split(x_values, y_values,indices, test_size=0.20,random_state=1)
x_train, x_val, y_train, y_val,indices_train,indices_val =train_test_split(x_train, y_train, indices_train, test_size=0.25,random_state=1)

x_train     = np.expand_dims(x_train, axis=2)
x_test      = np.expand_dims(x_test, axis=2)
x_val       =np.expand_dims(x_val, axis=2)

# %% MODEL
Epochs = 0 #300
batch_size = 100
lr = 0.001

#%% model
model=Sequential()
model.add(CuDNNLSTM(units=192, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(units=192, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(1)))
plot_model(model, to_file=R'C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Results_LSTM\simple_lstm_model_timedistestnormallr_withclipnorm_mae.png', show_shapes=True, show_layer_names=True)
model.summary()

opt = adam_v2.Adam(learning_rate=lr,clipnorm=1.0)

model.compile(optimizer=opt, loss='mean_absolute_error',metrics=['mae', 'mse'])

checkpoint_path = R"C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Results_LSTM\featureweights_lstm_simple_timedis_withclipnorm_mae.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Loads the weights
model.load_weights(checkpoint_path)

#Early stopping
#es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1)

start_time = time.time()
history = model.fit(x_train, y_train, batch_size=batch_size, 
                    epochs=Epochs,validation_data=(x_val,y_val),verbose=1,callbacks=[cp_callback]) #callbacks=[es,cp_callback])
print("--- %s seconds ---" % (time.time() - start_time))

#%% Val predictions
val_predictions = model.predict(x_val)


mse_train   = history.history['mse']
mse_val     = history.history['val_mse']
mae_train   = history.history['mae']
mae_val     = history.history['val_mae']

maetotal=mean_absolute_error(y_val,val_predictions[:,:,0])
msetotal=mean_squared_error(y_val,val_predictions[:,:,0])
rmsetotal=np.sqrt(msetotal)

plt.figure(figsize=(9, 3))
plt.subplot(121)
plt.plot(mae_train)
plt.plot(mae_val)
plt.title('Model MAE')
plt.ylim([0,25])
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
# summarize history for loss
plt.subplot(122)
plt.plot(mse_train)
plt.plot(mse_val)
plt.title('Model MSE')
plt.ylim([0,500])
plt.ylabel('Mean Squared Error')
plt.xlabel('Epochs')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


systolicknown=[]
systolicpredic=[]
diastolicknown=[]
diastolicpredic=[]
for x in range(len(val_predictions)):
    diastolicknown.append(min(y_val[x,:]))
    diastolicpredic.append(min(val_predictions[x,:]))
    systolicknown.append(max(y_val[x,:]))
    systolicpredic.append(max(val_predictions[x,:]))


differencedbp = []
differencesbp=[]
zip_objectdbp = zip(diastolicknown, diastolicpredic)
zip_objectsbp = zip(systolicknown, systolicpredic)

for diastolicknown_i, diastolicpredic_i in zip_objectdbp:
    differencedbp.append(diastolicknown_i-diastolicpredic_i)

for systolicknown_i, systolicpredic_i in zip_objectsbp:
    differencesbp.append(systolicknown_i-systolicpredic_i)

differencedbp=np.squeeze(differencedbp)
differencesbp=np.squeeze(differencesbp)
diastolicpredic=np.squeeze(diastolicpredic)
systolicpredic=np.squeeze(systolicpredic)

meanerrordbp=statistics.mean(differencedbp)
stdevdbp=np.std(differencedbp)

meanerrorsbp=statistics.mean(differencesbp)
stdevsbp=np.std(differencesbp)

print(meanerrordbp,meanerrorsbp,stdevdbp,stdevsbp)

[Rvaluedbp,pvaluedbp]= pearsonr(diastolicknown, diastolicpredic)

[Rvaluesbp,pvaluesbp]= pearsonr(systolicknown,systolicpredic)

yvalpredictionsdbp = np.vstack([diastolicknown,diastolicpredic])
zdbp = gaussian_kde(yvalpredictionsdbp)(yvalpredictionsdbp)

yvalpredictionssbp = np.vstack([systolicknown,systolicpredic])
zsbp = gaussian_kde(yvalpredictionssbp)(yvalpredictionssbp)

plt.figure()
plt.scatter(diastolicknown, diastolicpredic,c=zdbp,s=10) 
plt.plot([40, 140], [40, 140], '--k') 
plt.title('Correlation plot DBP validation set')
#plt.ylim([0,500])
plt.xlabel('Expected DBP')
plt.ylabel('Predictions DBP')

plt.figure()
plt.scatter(systolicknown, systolicpredic,c=zsbp,s=10) 
plt.plot([100, 200], [100, 200], '--k') 
plt.title('Correlation plot SBP validation set')
#plt.ylim([0,500])
plt.xlabel('Expected SBP')
plt.ylabel('Predictions SBP')

#Bland altman plot diastolic blood pressure
data1dbp     = diastolicknown
data2dbp     = diastolicpredic
meandbp      = np.mean([data1dbp, data2dbp], axis=0)
diffdbp      = data1dbp - data2dbp                   # Difference between data1 and data2
mddbp        = np.mean(diffdbp)                   # Mean of the difference
sddbp        = np.std(diffdbp, axis=0)            # Standard deviation of the difference

print('Mean difference diastolic blood pressure validation set:', mddbp)
print('Standard deviation of difference diastolic blood pressure validation set:',sddbp)
plt.figure()
plt.scatter(meandbp, diffdbp, c=zdbp,s=10)
plt.axhline(mddbp,           color='gray', linestyle='--')
plt.text(130,mddbp+2*sddbp, '+sd1.96', fontsize=10)
plt.axhline(mddbp + 1.96*sddbp, color='gray', linestyle='--')
plt.text(130,mddbp, 'Mean diff', fontsize=10)
plt.axhline(mddbp - 1.96*sddbp, color='gray', linestyle='--')
plt.text(130,mddbp-2*sddbp, '-sd1.96', fontsize=10)
plt.title('Bland-Altman Plot Diastolic Blood Pressure')
plt.show()

#Bland altman plot Systolic blood pressure
data1sbp     =systolicknown
data2sbp     = systolicpredic
meansbp      = np.mean([data1sbp, data2sbp], axis=0)
diffsbp      = data1sbp - data2sbp                   # Difference between data1 and data2
mdsbp        = np.mean(diffsbp)                   # Mean of the difference
sdsbp        = np.std(diffsbp, axis=0)            # Standard deviation of the difference

print('Mean difference systolic blood pressure validation set:', mdsbp)
print('Standard deviation of difference systolic blood pressure validation set:', sdsbp)
plt.figure()
plt.scatter(meansbp, diffsbp, c=zsbp,s=10)
plt.axhline(mdsbp,           color='gray', linestyle='--')
plt.text(50,mdsbp+2*sdsbp, '+sd1.96', fontsize=10)
plt.axhline(mdsbp + 1.96*sdsbp, color='gray', linestyle='--')
plt.text(50,mdsbp, 'Mean diff', fontsize=10)
plt.axhline(mdsbp - 1.96*sdsbp, color='gray', linestyle='--')
plt.text(50,mdsbp-2*sdsbp, '-sd1.96', fontsize=10)
plt.title('Bland-Altman Plot Systolic Blood Pressure')
plt.show()


totalamount=y_val.shape[0]
# BHS standards DBP
differencesdbp=np.abs(differencedbp)
listlower5dbp = [i for i in differencesdbp if i < 5]
listlower10dbp = [i for i in differencesdbp if i < 10]
listlower15dbp = [i for i in differencesdbp if i < 15]
percentage5dbp=(len(listlower5dbp)/totalamount)*100
percentage10dbp=(len(listlower10dbp)/totalamount)*100
percentage15dbp=(len(listlower15dbp)/totalamount)*100

# BHS standards SBP
differencessbp=np.abs(differencesbp)
listlower5sbp = [i for i in differencessbp if i < 5]
listlower10sbp = [i for i in differencessbp if i < 10]
listlower15sbp = [i for i in differencessbp if i < 15]
percentage5sbp=(len(listlower5sbp)/totalamount)*100
percentage10sbp=(len(listlower10sbp)/totalamount)*100
percentage15sbp=(len(listlower15sbp)/totalamount)*100

#BHS table
table=[['','<5 mmHg error [%]', '<10 mmHg error [%]', '<15 mmHg error [%]', 'BHS grade','R', 'Mean error','Standard dev'],['DBP',percentage5dbp,percentage10dbp,percentage15dbp,' ' ,Rvaluedbp,meanerrordbp,stdevdbp],['SBP',percentage5sbp,percentage10sbp,percentage15sbp, ' ',Rvaluesbp,meanerrorsbp,stdevsbp ]]
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

#BHS Histogram plot dbp
plt.figure()
plt.style.use('ggplot')
plt.hist(differencesdbp,bins=1,color='b')
plt.axvline(5, color='k', linestyle='dashed', linewidth=1)
plt.axvline(10, color='k', linestyle='dashed', linewidth=1)
plt.axvline(15, color='k', linestyle='dashed', linewidth=1)
plt.show()

#BHS Histogram plot sbp
plt.figure()
plt.style.use('ggplot')
plt.hist(differencessbp,bins=1,color='b')
plt.axvline(5, color='k', linestyle='dashed', linewidth=1)
plt.axvline(10, color='k', linestyle='dashed', linewidth=1)
plt.axvline(15, color='k', linestyle='dashed', linewidth=1)
plt.show()


#%% Test predictions
test_predictions = model.predict(x_test)

#Plot example sample 0
plt.figure()
plt.plot(y_test[100,:])
plt.plot(test_predictions[100,:,0])
plt.xlabel('Sample size [-]',fontsize=16)
plt.ylabel('Blood pressure [mmHg]',fontsize=16)
plt.legend(['Actual blood pressure', 'Predicted blood pressure'], loc='upper right',fontsize=14)


maetotaltest=mean_absolute_error(y_test,test_predictions[:,:,0])
msetotaltest=mean_squared_error(y_test,test_predictions[:,:,0])
rmsetotaltest=np.sqrt(msetotaltest)

systolicknowntest=[]
systolicpredictest=[]
diastolicknowntest=[]
diastolicpredictest=[]
for x in range(len(test_predictions)):
    diastolicknowntest.append(min(y_test[x,:]))
    diastolicpredictest.append(min(test_predictions[x,:]))
    systolicknowntest.append(max(y_test[x,:]))
    systolicpredictest.append(max(test_predictions[x,:]))


differencedbptest = []
differencesbptest=[]
zip_objectdbptest = zip(diastolicknowntest, diastolicpredictest)
zip_objectsbptest = zip(systolicknowntest, systolicpredictest)

for diastolicknown_itest, diastolicpredic_itest in zip_objectdbptest:
    differencedbptest.append(diastolicknown_itest-diastolicpredic_itest)

for systolicknown_itest, systolicpredic_itest in zip_objectsbptest:
    differencesbptest.append(systolicknown_itest-systolicpredic_itest)

differencedbptest=np.squeeze(differencedbptest)
differencesbptest=np.squeeze(differencesbptest)
diastolicpredictest=np.squeeze(diastolicpredictest)
systolicpredictest=np.squeeze(systolicpredictest)

meanerrordbptest=statistics.mean(differencedbptest)
stdevdbptest=np.std(differencedbptest)

meanerrorsbptest=statistics.mean(differencesbptest)
stdevsbptest=np.std(differencesbptest)

print(meanerrordbptest,meanerrorsbptest,stdevdbptest,stdevsbptest)

[Rvaluedbptest,pvaluedbptest]= pearsonr(diastolicknowntest, diastolicpredictest)

[Rvaluesbptest,pvaluesbptest]= pearsonr(systolicknowntest,systolicpredictest)

yvalpredictionsdbptest = np.vstack([diastolicknowntest,diastolicpredictest])
zdbptest = gaussian_kde(yvalpredictionsdbptest)(yvalpredictionsdbptest)

yvalpredictionssbptest = np.vstack([systolicknowntest,systolicpredictest])
zsbptest = gaussian_kde(yvalpredictionssbptest)(yvalpredictionssbptest)

plt.figure()
plt.scatter(diastolicknowntest, diastolicpredictest,c=zdbptest,s=10) 
plt.plot([40, 140], [40, 140], '--k') 
plt.title('Correlation plot Diastolic Blood Pressure')
#plt.ylim([0,500])
plt.xlabel('Actual DBP [mmHg]')
plt.ylabel('Predicted DBP [mmHg]')

plt.figure()
plt.scatter(systolicknowntest, systolicpredictest,c=zsbptest,s=10) 
plt.plot([80, 200], [80, 200], '--k') 
plt.title('Correlation plot Systolic Blood Pressure')
#plt.ylim([0,500])
plt.xlabel('Actual SBP [mmHg]')
plt.ylabel('Predicted SBP [mmHg]')

#Bland altman plot diastolic blood pressure
data1dbptest     = diastolicknowntest
data2dbptest     = diastolicpredictest
meandbptest      = np.mean([data1dbptest, data2dbptest], axis=0)
diffdbptest      = data1dbptest - data2dbptest                   # Difference between data1 and data2
mddbptest        = np.mean(diffdbptest)                   # Mean of the difference
sddbptest        = np.std(diffdbptest, axis=0)            # Standard deviation of the difference

print('Mean difference diastolic blood pressure test set:', mddbptest)
print('Standard deviation of difference diastolic blood pressure test set:',sddbptest)
plt.figure()
plt.scatter(meandbptest, diffdbptest, c=zdbptest,s=10)
plt.axhline(mddbptest,           color='gray', linestyle='--')
plt.text(82,mddbptest+1.96*sddbptest+0.4, '+sd1.96', fontsize=10)
plt.axhline(mddbptest + 1.96*sddbptest, color='gray', linestyle='--')
plt.text(82,mddbptest+0.4, 'Mean diff', fontsize=10)
plt.axhline(mddbptest - 1.96*sddbptest, color='gray', linestyle='--')
plt.text(82,mddbptest-1.96*sddbptest+0.4, '-sd1.96', fontsize=10)
plt.title('Bland-Altman Plot Diastolic Blood Pressure')
plt.xlabel('Diastolic Blood Pressure [mmHg]')
plt.ylabel('Error [mmHg]')
plt.show()

#Bland altman plot Systolic blood pressure
data1sbptest     =systolicknowntest
data2sbptest     = systolicpredictest
meansbptest      = np.mean([data1sbptest, data2sbptest], axis=0)
diffsbptest      = data1sbptest - data2sbptest                   # Difference between data1 and data2
mdsbptest        = np.mean(diffsbptest)                   # Mean of the difference
sdsbptest        = np.std(diffsbptest, axis=0)            # Standard deviation of the difference

print('Mean difference systolic blood pressure test set:', mdsbptest)
print('Standard deviation of difference systolic blood pressure test set:', sdsbptest)
plt.figure()
plt.scatter(meansbptest, diffsbptest, c=zsbptest,s=10)
plt.axhline(mdsbptest,           color='gray', linestyle='--')
plt.text(80,mdsbptest+1.96*sdsbptest+0.4, '+sd1.96', fontsize=10)
plt.axhline(mdsbptest + 1.96*sdsbptest, color='gray', linestyle='--')
plt.text(80,mdsbptest+0.4, 'Mean diff', fontsize=10)
plt.axhline(mdsbptest - 1.96*sdsbptest, color='gray', linestyle='--')
plt.text(80,mdsbptest-1.96*sdsbptest+0.3, '-sd1.96', fontsize=10)
plt.title('Bland-Altman Plot Systolic Blood Pressure')
plt.xlabel('Systolic Blood Pressure [mmHg]')
plt.ylabel('Error [mmHg]')
plt.show()


totalamounttest=y_test.shape[0]
# BHS standards DBP
differencesdbptest=np.abs(differencedbptest)
listlower5dbptest = [itest for itest in differencesdbptest if itest < 5]
listlower10dbptest = [itest for itest in differencesdbptest if itest < 10]
listlower15dbptest = [itest for itest in differencesdbptest if itest < 15]
percentage5dbptest=(len(listlower5dbptest)/totalamounttest)*100
percentage10dbptest=(len(listlower10dbptest)/totalamounttest)*100
percentage15dbptest=(len(listlower15dbptest)/totalamounttest)*100

# BHS standards SBP
differencessbptest=np.abs(differencesbptest)
listlower5sbptest = [itest for itest in differencessbptest if itest < 5]
listlower10sbptest = [itest for itest in differencessbptest if itest < 10]
listlower15sbptest = [itest for itest in differencessbptest if itest < 15]
percentage5sbptest=(len(listlower5sbptest)/totalamounttest)*100
percentage10sbptest=(len(listlower10sbptest)/totalamounttest)*100
percentage15sbptest=(len(listlower15sbptest)/totalamounttest)*100

#BHS table
table=[['','<5 mmHg error [%]', '<10 mmHg error [%]', '<15 mmHg error [%]', 'BHS grade','R', 'Mean error','Standard dev'],['DBP',percentage5dbptest,percentage10dbptest,percentage15dbptest,' ' ,Rvaluedbptest,meanerrordbptest,stdevdbptest],['SBP',percentage5sbptest,percentage10sbptest,percentage15sbptest, ' ',Rvaluesbptest,meanerrorsbptest,stdevsbptest ]]
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

q25testdbp,q75testdbp=np.percentile(differencesdbptest,[25, 75])
bin_widthdbp=2*(q75testdbp - q25testdbp)*len(differencesdbptest)**(-1/3)
binsdbp=round((differencesdbptest.max()-differencesdbptest.min())/bin_widthdbp)

q25testsbp,q75testsbp=np.percentile(differencessbptest,[25, 75])
bin_widthsbp=2*(q75testsbp - q25testsbp)*len(differencessbptest)**(-1/3)
binssbp=round((differencessbptest.max()-differencessbptest.min())/bin_widthsbp)
#BHS Histogram plot dbp
plt.figure()
plt.style.use('ggplot')
plt.hist(differencesdbptest,bins=binsdbp,color='b')
plt.axvline(5, color='k', linestyle='dashed', linewidth=1)
plt.axvline(10, color='k', linestyle='dashed', linewidth=1)
plt.axvline(15, color='k', linestyle='dashed', linewidth=1)
plt.title('BHS grading histogram Diastolic Blood Pressure')
plt.xlabel('Error [mmHg]')
plt.ylabel('Count')
plt.show()

#BHS Histogram plot sbp
plt.figure()
plt.style.use('ggplot')
plt.hist(differencessbptest,bins=binssbp,color='b')
plt.axvline(5, color='k', linestyle='dashed', linewidth=1)
plt.axvline(10, color='k', linestyle='dashed', linewidth=1)
plt.axvline(15, color='k', linestyle='dashed', linewidth=1)
plt.title('BHS grading histogram Systolic Blood Pressure')
plt.xlabel('Error [mmHg]')
plt.ylabel('Count')
plt.show()

# %%
