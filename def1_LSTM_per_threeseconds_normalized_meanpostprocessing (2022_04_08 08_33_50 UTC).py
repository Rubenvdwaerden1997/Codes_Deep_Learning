from keras.utils.vis_utils import plot_model
import numpy as np
from numpy import *
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
from scipy.signal import find_peaks


# %% DATA
os.chdir(R'C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Matlab\FINAL_preparation_ML_models\CNN_400_Persons')
inputall = loadmat('lstm_data_per_timescale_normalized.mat')  

x_values=inputall['x_values']
y_values=inputall['y_values']
indices=range(len(x_values))
del inputall

x_train, x_test, y_train, y_test,indices_train,indices_test  =train_test_split(x_values, y_values,indices, test_size=0.20,random_state=1)
x_train, x_val, y_train, y_val,indices_train,indices_val =train_test_split(x_train, y_train, indices_train, test_size=0.25,random_state=1)

x_train     = np.expand_dims(x_train, axis=2)
x_test      = np.expand_dims(x_test, axis=2)
x_val       =np.expand_dims(x_val, axis=2)

# Table verdeling dbp en sbp
# #%% Systolic and diastolic calculations test set
# sbpvaluesknown=[]
# dbpvaluesknown=[]

# for i in range(y_values.shape[0]):
#     knownbptest=y_values[i,:]
#     peaksknowntest, _ = find_peaks(knownbptest, distance=60, height=77) #Dus ligt 30 plekken tussen pieken en minimaal hoogte van 77
#     valleysknowntest, _ = find_peaks(-knownbptest, distance=60, height=-150) #Dus ligt 30 plekken tussen pieken en minimaal hoogte van -130
#     if len(valleysknowntest) == 0 or len(peaksknowntest) == 0 :
#         print('the list is empty')
#         continue
#     meanvalueknownsbptest=mean(knownbptest[peaksknowntest])
#     meanvalueknowndbptest=mean(knownbptest[valleysknowntest])
#     sbpvaluesknown.append(meanvalueknownsbptest)
#     dbpvaluesknown.append(meanvalueknowndbptest)

# #Dataset statistics
# mindbp=min(dbpvaluesknown)
# minsbp=min(sbpvaluesknown)
# maxdbp=max(dbpvaluesknown)
# maxsbp=max(sbpvaluesknown)
# meandbp=np.mean(dbpvaluesknown)
# meansbp=np.mean(sbpvaluesknown)
# stddbp=np.std(dbpvaluesknown)
# stdsbp=np.std(sbpvaluesknown)
# samplesize=len(x_values)

# tabelstatsbloodpressure=[' ','Min', 'Max', 'Mean', 'Stdev', 'Sample size'], ['DBP', mindbp,maxdbp,meandbp,stddbp,samplesize], ['SBP', minsbp,maxsbp,meansbp,stdsbp,samplesize]
# print(tabulate(tabelstatsbloodpressure, headers='firstrow', tablefmt='fancy_grid'))

# %% MODEL
Epochs = 0 #250
batch_size = 100 #100
lr = 0.001#0.001

#%% model
model=Sequential()
model.add(CuDNNLSTM(units=192, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(units=192, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(1)))
plot_model(model, to_file=R'C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\ML_results\Def1_LSTM_per_3sec\lstm_normalized_3seconds.png', show_shapes=True, show_layer_names=True)
model.summary()

opt = adam_v2.Adam(learning_rate=lr,clipnorm=1.0)

model.compile(optimizer=opt, loss='mean_absolute_error',metrics=['mae', 'mse'])

checkpoint_path = R"C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\ML_results\Def1_LSTM_per_3sec\weights_timescale_normalized_loss_mae_after_analysis.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

# Loads the weights
model.load_weights(checkpoint_path)


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


# mse_train   = history.history['mse']
# mse_val     = history.history['val_mse']
# mae_train   = history.history['mae']
# mae_val     = history.history['val_mae']

maetotal=mean_absolute_error(y_val,val_predictions[:,:,0])
msetotal=mean_squared_error(y_val,val_predictions[:,:,0])
rmsetotal=np.sqrt(msetotal)

# plt.figure(figsize=(9, 3))
# plt.subplot(121)
# plt.plot(mae_train)
# plt.plot(mae_val)
# plt.title('Model MAE')
# plt.ylim([0,25])
# plt.ylabel('Mean Absolute Error')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='upper left')
# # summarize history for loss
# plt.subplot(122)
# plt.plot(mse_train)
# plt.plot(mse_val)
# plt.title('Model MSE')
# plt.ylim([0,500])
# plt.ylabel('Mean Squared Error')
# plt.xlabel('Epochs')
# plt.legend(['Train', 'Val'], loc='upper left')
# plt.show()




#%% Systolic and diastolic calculations
systolicknown=[]
systolicpredic=[]
diastolicknown=[]
diastolicpredic=[]

differencesbp=[]
differencedbp=[]
for i in range(val_predictions.shape[0]):
    estimbp=val_predictions[i,:,0]
    knownbp=y_val[i,:]
    peaksest, _ = find_peaks(estimbp, distance=60, height=77) #Dus ligt 30 plekken tussen pieken en minimaal hoogte van 80
    peaksknown, _ = find_peaks(knownbp, distance=60, height=77) #Dus ligt 30 plekken tussen pieken en minimaal hoogte van 80
    valleysest, _ = find_peaks(-estimbp, distance=60, height=-130) #Dus ligt 30 plekken tussen pieken en minimaal hoogte van 80
    valleysknown, _ = find_peaks(-knownbp, distance=60, height=-130) #Dus ligt 30 plekken tussen pieken en minimaal hoogte van 80
    meanvalueestsbp=mean(estimbp[peaksest])
    meanvalueknownsbp=mean(knownbp[peaksknown])
    meanvalueestdbp=mean(estimbp[valleysest])
    meanvalueknowndbp=mean(knownbp[valleysknown])
    systolicknown.append(meanvalueknownsbp)
    systolicpredic.append(meanvalueestsbp)
    diastolicknown.append(meanvalueknowndbp)
    diastolicpredic.append(meanvalueestdbp)
    diffvalsbp=meanvalueknownsbp-meanvalueestsbp
    differencesbp.append(diffvalsbp)
    diffvaldbp=meanvalueknowndbp-meanvalueestdbp
    differencedbp.append(diffvaldbp)



# For plotting de pieken en de bijbehorende bekende en geschatte bloeddruk
# plt.figure()
# plt.plot(peaksest,estimbp[peaksest],'x')
# plt.plot(val_predictions[i,:,0])
# plt.plot(peaksknown,knownbp[peaksknown],'x')
# plt.plot(y_val[i,:])


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
plt.plot([75, 200], [75, 200], '--k') 
plt.title('Correlation plot SBP validation set')
#plt.ylim([0,500])
plt.xlabel('Expected SBP')
plt.ylabel('Predictions SBP')

#Bland altman plot diastolic blood pressure
data1dbp     = diastolicknown
data2dbp     = diastolicpredic
meandbp      = np.mean([data1dbp, data2dbp], axis=0)
diffdbp      = [a_i - b_i for a_i, b_i in zip(data1dbp, data2dbp)]                   # Difference between data1 and data2
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
plt.xlabel('Diastolic Blood Pressure [mmHg]')
plt.ylabel('Error [mmHg]')
plt.show()

#Bland altman plot Systolic blood pressure
data1sbp     =systolicknown
data2sbp     = systolicpredic
meansbp      = np.mean([data1sbp, data2sbp], axis=0)
diffsbp      = [a_i - b_i for a_i, b_i in zip(data1sbp, data2sbp)]                  # Difference between data1 and data2
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
plt.xlabel('Systolic Blood Pressure [mmHg]')
plt.ylabel('Error [mmHg]')
plt.show()


totalamountdbp=len(data1dbp)
totalamountsbp=len(data1sbp)
# BHS standards DBP
differencesdbp=np.abs(differencedbp)
listlower5dbp = [i for i in differencesdbp if i < 5]
listlower10dbp = [i for i in differencesdbp if i < 10]
listlower15dbp = [i for i in differencesdbp if i < 15]
percentage5dbp=(len(listlower5dbp)/totalamountdbp)*100
percentage10dbp=(len(listlower10dbp)/totalamountdbp)*100
percentage15dbp=(len(listlower15dbp)/totalamountdbp)*100

# BHS standards SBP
differencessbp=np.abs(differencesbp)
listlower5sbp = [i for i in differencessbp if i < 5]
listlower10sbp = [i for i in differencessbp if i < 10]
listlower15sbp = [i for i in differencessbp if i < 15]
percentage5sbp=(len(listlower5sbp)/totalamountsbp)*100
percentage10sbp=(len(listlower10sbp)/totalamountsbp)*100
percentage15sbp=(len(listlower15sbp)/totalamountsbp)*100

#BHS table
table=[['','<5 mmHg error [%]', '<10 mmHg error [%]', '<15 mmHg error [%]', 'BHS grade','R', 'Mean error','Standard dev'],['DBP',percentage5dbp,percentage10dbp,percentage15dbp,' ' ,Rvaluedbp,meanerrordbp,stdevdbp],['SBP',percentage5sbp,percentage10sbp,percentage15sbp, ' ',Rvaluesbp,meanerrorsbp,stdevsbp ]]
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))


q25dbp,q75dbp=np.percentile(differencesdbp,[25, 75])
bin_widthdbp=2*(q75dbp - q25dbp)*len(differencesdbp)**(-1/3)
binsdbp=round((differencesdbp.max()-differencesdbp.min())/bin_widthdbp)

q25sbp,q75sbp=np.percentile(differencessbp,[25, 75])
bin_widthsbp=2*(q75sbp - q25sbp)*len(differencessbp)**(-1/3)
binssbp=round((differencessbp.max()-differencessbp.min())/bin_widthsbp)

#BHS Histogram plot dbp
plt.figure()
plt.style.use('ggplot')
plt.hist(differencesdbp,bins=binsdbp,color='b')
plt.axvline(5, color='k', linestyle='dashed', linewidth=1)
plt.axvline(10, color='k', linestyle='dashed', linewidth=1)
plt.axvline(15, color='k', linestyle='dashed', linewidth=1)
plt.title('BHS grading histogram Diastolic Blood Pressure')
plt.show()

#BHS Histogram plot sbp
plt.figure()
plt.style.use('ggplot')
plt.hist(differencessbp,bins=binssbp,color='b')
plt.axvline(5, color='k', linestyle='dashed', linewidth=1)
plt.axvline(10, color='k', linestyle='dashed', linewidth=1)
plt.axvline(15, color='k', linestyle='dashed', linewidth=1)
plt.title('BHS grading histogram Systolic Blood Pressure')
plt.show()
# %%

#%% predictions
test_predictions = model.predict(x_test)


#Plot example sample 1
timelength=np.linspace(0, 3, num=375)
plt.figure()
plt.plot(timelength,y_test[0,:])
plt.plot(timelength,test_predictions[0,:,0])
plt.xlabel('Time [s]')
plt.ylabel('Blood pressure [mmHg]')
plt.legend(['Actual blood pressure', 'Predicted blood pressure'], loc='upper right')

maetotaltest=mean_absolute_error(y_test,test_predictions[:,:,0])
msetotaltest=mean_squared_error(y_test,test_predictions[:,:,0])
rmsetotaltest=np.sqrt(msetotaltest)


#%% Systolic and diastolic calculations test set
systolicknowntest=[]
systolicpredictest=[]
diastolicknowntest=[]
diastolicpredictest=[]

differencesbptest=[]
differencedbptest=[]
for i in range(test_predictions.shape[0]):
    estimbptest=test_predictions[i,:,0]
    knownbptest=y_test[i,:]
    peaksesttest, _ = find_peaks(estimbptest, distance=60, height=77) #Dus ligt 30 plekken tussen pieken en minimaal hoogte van 77
    peaksknowntest, _ = find_peaks(knownbptest, distance=60, height=77) #Dus ligt 30 plekken tussen pieken en minimaal hoogte van 77
    valleysesttest, _ = find_peaks(-estimbptest, distance=60, height=-150) #Dus ligt 30 plekken tussen pieken en minimaal hoogte van -130
    valleysknowntest, _ = find_peaks(-knownbptest, distance=60, height=-150) #Dus ligt 30 plekken tussen pieken en minimaal hoogte van -130
    if len(valleysknowntest) == 0 or len(valleysesttest)==0 or len(peaksknowntest) == 0 or len(peaksesttest)==0:
        print('the list is empty')
        continue
    meanvalueestsbptest=mean(estimbptest[peaksesttest])
    meanvalueknownsbptest=mean(knownbptest[peaksknowntest])
    meanvalueestdbptest=mean(estimbptest[valleysesttest])
    meanvalueknowndbptest=mean(knownbptest[valleysknowntest])
    systolicknowntest.append(meanvalueknownsbptest)
    systolicpredictest.append(meanvalueestsbptest)
    diastolicknowntest.append(meanvalueknowndbptest)
    diastolicpredictest.append(meanvalueestdbptest)
    diffvalsbptest=meanvalueknownsbptest-meanvalueestsbptest
    differencesbptest.append(diffvalsbptest)
    diffvaldbptest=meanvalueknowndbptest-meanvalueestdbptest
    differencedbptest.append(diffvaldbptest)

meanerrordbptest=statistics.mean(differencedbptest)
stdevdbptest=np.std(differencedbptest)

meanerrorsbptest=np.mean(differencesbptest)
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
plt.title('Correlation plot Diastolic Blood Pressure ')
plt.ylabel('Predicted DBP [mmHg]')
plt.xlabel('Actual DBP [mmHg]')

plt.figure()
plt.scatter(systolicknowntest, systolicpredictest,c=zsbptest,s=10) 
plt.plot([75, 200], [75, 200], '--k') 
plt.title('Correlation plot Systolic Blood Pressure')
plt.ylabel('Predicted SBP [mmHg]')
plt.xlabel('Actual SBP [mmHg]')

#Bland altman plot diastolic blood pressure
data1dbptest     = diastolicknowntest
data2dbptest     = diastolicpredictest
meandbptest      = np.mean([data1dbptest, data2dbptest], axis=0)
diffdbptest      = [a_i - b_i for a_i, b_i in zip(data1dbptest, data2dbptest)]                   # Difference between data1 and data2
mddbptest        = np.mean(diffdbptest)                   # Mean of the difference
sddbptest        = np.std(diffdbptest, axis=0)            # Standard deviation of the difference

print('Mean difference diastolic blood pressure test set:', mddbptest)
print('Standard deviation of difference diastolic blood pressure test set:',sddbptest)
plt.figure()
plt.scatter(meandbptest, diffdbptest, c=zdbptest,s=10)
plt.axhline(mddbptest,           color='gray', linestyle='--')
plt.text(105,mddbptest+1.96*sddbptest+0.4, '+sd1.96', fontsize=10)
plt.axhline(mddbptest + 1.96*sddbptest, color='gray', linestyle='--')
plt.text(105,mddbptest, 'Mean diff', fontsize=10)
plt.axhline(mddbptest - 1.96*sddbptest, color='gray', linestyle='--')
plt.text(105,mddbptest-1.96*sddbptest+0.4, '-sd1.96', fontsize=10)
plt.title('Bland-Altman Plot Diastolic Blood Pressure')
plt.xlabel('Diastolic Blood Pressure [mmHg]')
plt.ylabel('Error [mmHg]')
plt.show()

#Bland altman plot Systolic blood pressure
data1sbptest     =systolicknowntest
data2sbptest     = systolicpredictest
meansbptest      = np.mean([data1sbptest, data2sbptest], axis=0)
diffsbptest      = [a_i - b_i for a_i, b_i in zip(data1sbptest, data2sbptest)]                  # Difference between data1 and data2
mdsbptest        = np.mean(diffsbptest)                   # Mean of the difference
sdsbptest        = np.std(diffsbptest, axis=0)            # Standard deviation of the difference

print('Mean difference systolic blood pressure validation set:', mdsbptest)
print('Standard deviation of difference systolic blood pressure validation set:', sdsbptest)
plt.figure()
plt.scatter(meansbptest, diffsbptest, c=zsbptest,s=10)
plt.axhline(mdsbptest,           color='gray', linestyle='--')
plt.text(88,mdsbptest+1.96*sdsbptest+0.4, '+sd1.96', fontsize=10)
plt.axhline(mdsbptest + 1.96*sdsbptest, color='gray', linestyle='--')
plt.text(88,mdsbptest, 'Mean diff', fontsize=10)
plt.axhline(mdsbptest - 1.96*sdsbptest, color='gray', linestyle='--')
plt.text(88,mdsbptest-1.96*sdsbptest+0.4, '-sd1.96', fontsize=10)
plt.title('Bland-Altman Plot Systolic Blood Pressure')
plt.xlabel('Systolic Blood Pressure [mmHg]')
plt.ylabel('Error [mmHg]')
plt.show()


totalamountdbptest=len(data1dbptest)
totalamountsbptest=len(data1sbptest)
# BHS standards DBP
differencesdbptest=np.abs(differencedbptest)
listlower5dbptest = [i for i in differencesdbptest if i < 5]
listlower10dbptest = [i for i in differencesdbptest if i < 10]
listlower15dbptest = [i for i in differencesdbptest if i < 15]
percentage5dbptest=(len(listlower5dbptest)/totalamountdbptest)*100
percentage10dbptest=(len(listlower10dbptest)/totalamountdbptest)*100
percentage15dbptest=(len(listlower15dbptest)/totalamountdbptest)*100

# BHS standards SBP
differencessbptest=np.abs(differencesbptest)
listlower5sbptest = [i for i in differencessbptest if i < 5]
listlower10sbptest = [i for i in differencessbptest if i < 10]
listlower15sbptest = [i for i in differencessbptest if i < 15]
percentage5sbptest=(len(listlower5sbptest)/totalamountsbptest)*100
percentage10sbptest=(len(listlower10sbptest)/totalamountsbptest)*100
percentage15sbptest=(len(listlower15sbptest)/totalamountsbptest)*100


#BHS table
tabletest=[['','<5 mmHg error [%]', '<10 mmHg error [%]', '<15 mmHg error [%]', 'BHS grade','R', 'Mean error','Standard dev'],['DBP',percentage5dbptest,percentage10dbptest,percentage15dbptest,' ' ,Rvaluedbptest,meanerrordbptest,stdevdbptest],['SBP',percentage5sbptest,percentage10sbptest,percentage15sbptest, ' ',Rvaluesbptest,meanerrorsbptest,stdevsbptest ]]
print(tabulate(tabletest, headers='firstrow', tablefmt='fancy_grid'))


q25dbptest,q75dbptest=np.percentile(differencesdbptest,[25, 75])
bin_widthdbptest=2*(q75dbptest - q25dbptest)*len(differencesdbptest)**(-1/3)
binsdbptest=round((differencesdbptest.max()-differencesdbptest.min())/bin_widthdbptest)

q25sbptest,q75sbptest=np.percentile(differencessbptest,[25, 75])
bin_widthsbptest=2*(q75sbptest - q25sbptest)*len(differencessbptest)**(-1/3)
binssbptest=round((differencessbptest.max()-differencessbptest.min())/bin_widthsbptest)


#BHS Histogram plot dbp
plt.figure()
plt.style.use('ggplot')
plt.hist(differencesdbptest,bins=binsdbptest,color='b')
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
plt.hist(differencessbptest,bins=binssbptest,color='b')
plt.axvline(5, color='k', linestyle='dashed', linewidth=1)
plt.axvline(10, color='k', linestyle='dashed', linewidth=1)
plt.axvline(15, color='k', linestyle='dashed', linewidth=1)
plt.title('BHS grading histogram Systolic Blood Pressure')
plt.xlabel('Error [mmHg]')
plt.ylabel('Count')
plt.show()