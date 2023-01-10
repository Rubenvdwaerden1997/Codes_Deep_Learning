## Load part needs to be on when you want to load an already pre trained model. The weights are loaded, so the same network shape is needed.
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling1D, Conv1D, BatchNormalization
import os
import tensorflow as tf
import time
from scipy.io import loadmat
import scipy.io as sio
import pygraphviz
import pydot
from keras.utils.vis_utils import plot_model
import statistics
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from tabulate import tabulate
from scipy.stats import pearsonr
from keras.models import Model


print('Modules imported')

# %% DATA
os.chdir(R'C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\Matlab\FINAL_preparation_ML_models\CNN_400_Persons')
inputall = loadmat('cnndata.mat')  

x_values=inputall['x_values']
y_values=inputall['ysdbp_values']
indices=range(len(x_values))
del inputall
x_train, x_test, y_train, y_test,indices_train,indices_test  =train_test_split(x_values, y_values,indices, test_size=0.20,random_state=1)
x_train, x_val, y_train, y_val,indices_train,indices_val =train_test_split(x_train, y_train, indices_train, test_size=0.25,random_state=1)

x_train     = np.expand_dims(x_train, axis=2)
x_test      = np.expand_dims(x_test, axis=2)
x_val       =np.expand_dims(x_val, axis=2)
#Dataset statistics
mindbp=min(y_values[:,0])
minsbp=min(y_values[:,1])
minmap=min(y_values[:,2])
maxdbp=max(y_values[:,0])
maxsbp=max(y_values[:,1])
maxmap=max(y_values[:,2])
meandbp=np.mean(y_values[:,0])
meansbp=np.mean(y_values[:,1])
meanmap=np.mean(y_values[:,2])
stddbp=np.std(y_values[:,0])
stdsbp=np.std(y_values[:,1])
stdmap=np.std(y_values[:,2])
samplesize=len(x_values)

tabelstatsbloodpressure=[' ','Min', 'Max', 'Mean', 'Stdev', 'Sample size'], ['DBP', mindbp,maxdbp,meandbp,stddbp,samplesize], ['SBP', minsbp,maxsbp,meansbp,stdsbp,samplesize],['MAP', minmap,maxmap,meanmap,stdmap,samplesize]
print(tabulate(tabelstatsbloodpressure, headers='firstrow', tablefmt='fancy_grid'))

# %% MODEL
Epochs = 0 #100
batch_size = 100
lr = 0.0001
Filters= 48
kernel = 19

#create model
model = Sequential()
#add model layers
model.add(Conv1D(Filters, kernel_size=kernel,padding='same', activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(Conv1D(Filters, kernel_size=kernel,padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=(2),padding='same'))
model.add(Conv1D(Filters*2, kernel_size=kernel,padding='same', activation='relu'))
model.add(Conv1D(Filters*2, kernel_size=kernel,padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=(2),padding='same'))
model.add(Conv1D(Filters, kernel_size=kernel,padding='same', activation='relu'))
model.add(Conv1D(Filters, kernel_size=kernel,padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=(2),padding='same'))
model.add(Conv1D(Filters/2, kernel_size=kernel,padding='same', activation='relu'))
model.add(Conv1D(Filters/2, kernel_size=kernel,padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=(2),padding='same'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(3))
#plot_model(model, to_file=R'C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\ML_results\Def1_CNN\MAE_loss\model_plot_ppg2diastolic_systolic_maptest.png', show_shapes=True, show_layer_names=True)

#summary of model
model.summary()

#compile model using mse to measure model performance
model.compile(optimizer='adam',
              loss='mae', 
              metrics=['mse','mae'])

checkpoint_path = R"C:\Users\s164052\Documents\Medical_Engineering_1\Afstuderen_Q4\ML_results\Def1_CNN\Def1_MAE_loss\cpfeaturestestdias_sys_map100_mae.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#evaluate model before loading the weights
scores = model.evaluate(x_train,y_train, verbose=2)
print("%s: %.2f" % (model.metrics_names[2], scores[2]))

# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
scores = model.evaluate(x_train,y_train, verbose=2)
print("%s: %.2f" % (model.metrics_names[2], scores[2]))


# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

            
#train model
start_time = time.time()
history=model.fit(x_train, y_train, batch_size=batch_size, 
                    epochs=Epochs,validation_data=(x_val,y_val), verbose=1,callbacks=[cp_callback])#, callbacks=[es])
print("--- %s seconds ---" % (time.time() - start_time))

scores = model.evaluate(x_val,y_val, verbose=1)
print("%s: %.2f" % (model.metrics_names[2], scores[2]))

# %% Validation predictions
val_predictions = model.predict(x_val)
train_predictions = model.predict(x_train)

scorestrain=model.evaluate(x_train,y_train,verbose=2)
scoresval=model.evaluate(x_val,y_val,verbose=2)

mse_train   = history.history['mse']
mse_val     = history.history['val_mse']
mae_train   = history.history['mae']
mae_val     = history.history['val_mae']

meanerrordbp=statistics.mean(y_val[:,0]-val_predictions[:,0])
stdevdbp=statistics.stdev(y_val[:,0]-val_predictions[:,0])

meanerrorsbp=statistics.mean(y_val[:,1]-val_predictions[:,1])
stdevsbp=statistics.stdev(y_val[:,1]-val_predictions[:,1])

meanerrormap=statistics.mean(y_val[:,2]-val_predictions[:,2])
stdevmap=statistics.stdev(y_val[:,2]-val_predictions[:,2])

print('validation set diastolic error and stdev:',meanerrordbp,stdevdbp)
print('validation set systolic error and stdev:',meanerrorsbp,stdevsbp)
print('validation set map error and stdev:',meanerrormap,stdevmap)
# %% LOSS PLOT
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
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()



totalamount=y_val.shape[0]
# BHS standards DBP
differencesdbp=abs(val_predictions[:,0]-y_val[:,0])
listlower5dbp = [i for i in differencesdbp if i < 5]
listlower10dbp = [i for i in differencesdbp if i < 10]
listlower15dbp = [i for i in differencesdbp if i < 15]
percentage5dbp=(len(listlower5dbp)/totalamount)*100
percentage10dbp=(len(listlower10dbp)/totalamount)*100
percentage15dbp=(len(listlower15dbp)/totalamount)*100


# BHS standards SBP
differencessbp=abs(val_predictions[:,1]-y_val[:,1])
listlower5sbp = [i for i in differencessbp if i < 5]
listlower10sbp = [i for i in differencessbp if i < 10]
listlower15sbp = [i for i in differencessbp if i < 15]
percentage5sbp=(len(listlower5sbp)/totalamount)*100
percentage10sbp=(len(listlower10sbp)/totalamount)*100
percentage15sbp=(len(listlower15sbp)/totalamount)*100

# BHS standards MAP
differencesmap=abs(val_predictions[:,2]-y_val[:,2])
listlower5map = [i for i in differencesmap if i < 5]
listlower10map = [i for i in differencesmap if i < 10]
listlower15map = [i for i in differencesmap if i < 15]
percentage5map=(len(listlower5map)/totalamount)*100
percentage10map=(len(listlower10map)/totalamount)*100
percentage15map=(len(listlower15map)/totalamount)*100


yvalpredictionsdbp = np.vstack([y_val[:,0],val_predictions[:,0]])
zdbp = gaussian_kde(yvalpredictionsdbp)(yvalpredictionsdbp)

plt.figure()
plt.scatter(y_val[:,0], val_predictions[:,0],c=zdbp,s=10) 
plt.plot([40, 140], [40, 140], '--k') 
plt.title('Correlation plot DBP validation set')
#plt.ylim([0,500])
plt.xlabel('Expected DBP')
plt.ylabel('Predictions DBP')

yvalpredictionssbp = np.vstack([y_val[:,1],val_predictions[:,1]])
zsbp = gaussian_kde(yvalpredictionssbp)(yvalpredictionssbp)
plt.figure()
plt.scatter(y_val[:,1], val_predictions[:,1],c=zsbp,s=10) 
plt.plot([80, 200], [80, 200], '--k') 
plt.title('Correlation plot SBP validation set')
#plt.ylim([0,500])
plt.xlabel('Expected SBP')
plt.ylabel('Predictions SBP')

yvalpredictionsmap = np.vstack([y_val[:,2],val_predictions[:,2]])
zmap = gaussian_kde(yvalpredictionsmap)(yvalpredictionsmap)
plt.figure()
plt.scatter(y_val[:,2], val_predictions[:,2],c=zmap,s=10) 
plt.plot([60, 160], [60, 160], '--k') 
plt.title('Correlation plot map validation set')
#plt.ylim([0,500])
plt.xlabel('Expected map')
plt.ylabel('Predictions map')

# Correlation coefficient
[Rvaluedbp,pvaluedbp]= pearsonr(y_val[:,0], val_predictions[:,0])

[Rvaluesbp,pvaluesbp]= pearsonr(y_val[:,1],val_predictions[:,1])

[Rvaluemap,pvaluemap]= pearsonr(y_val[:,2],val_predictions[:,2])

print('Pearsons correlation diastolic blood pressure validation set: %.3f' % Rvaluedbp)
print('Pearsons correlation systolic blood pressure validation set: %.3f' % Rvaluesbp)
print('Pearsons correlation mean arterial blood pressure validation set: %.3f' % Rvaluemap)



#Bland altman plot diastolic blood pressure
data1     = np.asarray(y_val[:,0])
data2     = np.asarray(val_predictions[:,0])
mean      = np.mean([data1, data2], axis=0)
diff      = data1 - data2                   # Difference between data1 and data2
md        = np.mean(diff)                   # Mean of the difference
sd        = np.std(diff, axis=0)            # Standard deviation of the difference

print('Mean difference diastolic blood pressure validation set:', md)
print('Standard deviation of difference diastolic blood pressure validation set:',sd)
plt.figure()
plt.scatter(mean, diff, c=zdbp,s=10)
plt.axhline(md,           color='gray', linestyle='--')
plt.text(130,md+2*sd, '+sd1.96', fontsize=10)
plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
plt.text(130,md, 'Mean diff', fontsize=10)
plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
plt.text(130,md-2*sd, '-sd1.96', fontsize=10)
plt.title('Bland-Altman Plot Diastolic Blood Pressure')
plt.show()

idxdbp = (np.abs(diff - 0)).argmin()
bestpreddbp=indices_val[idxdbp]
maxerrordbp=np.where(diff==max(diff))[0][0] #Dit is maximale positieve error
minerrordbp=np.where(diff==min(diff))[0][0] #Dit is maximale negatieve error
indexmaxerror=indices_val[maxerrordbp]
indexminerror=indices_val[minerrordbp]
plt.figure()
plt.plot(x_values[indexmaxerror,:])
plt.plot(x_values[indexminerror,:])
plt.plot(x_values[bestpreddbp,:])
plt.title('PPG signals with highest and lowest error diastolic')
plt.legend(['PPG signal with highest positive diastolic error', 'PPG signal with highest negative diastolic error', 'PPG signal with lowest error (closest to 0)'])

#Bland altman plot Systolic blood pressure
data1     = np.asarray(y_val[:,1])
data2     = np.asarray(val_predictions[:,1])
mean      = np.mean([data1, data2], axis=0)
diff      = data1 - data2                   # Difference between data1 and data2
md        = np.mean(diff)                   # Mean of the difference
sd        = np.std(diff, axis=0)            # Standard deviation of the difference

print('Mean difference systolic blood pressure validation set:', md)
print('Standard deviation of difference systolic blood pressure validation set:', sd)
plt.figure()
plt.scatter(mean, diff, c=zsbp,s=10)
plt.axhline(md,           color='gray', linestyle='--')
plt.text(50,md+2*sd, '+sd1.96', fontsize=10)
plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
plt.text(50,md, 'Mean diff', fontsize=10)
plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
plt.text(50,md-2*sd, '-sd1.96', fontsize=10)
plt.title('Bland-Altman Plot Systolic Blood Pressure')
plt.show()
# %% Find outliers

idxsbp = (np.abs(diff - 0)).argmin()
bestpredsbp=indices_val[idxsbp]
maxerrorsbp=np.where(diff==max(diff))[0][0]
minerrorsbp=np.where(diff==min(diff))[0][0]
indexmaxerror=indices_val[maxerrorsbp]
indexminerror=indices_val[minerrorsbp]
plt.figure()
plt.plot(x_values[indexmaxerror,:])
plt.plot(x_values[indexminerror,:])
plt.plot(x_values[bestpredsbp,:])
plt.title('PPG signals with highest and lowest error systolic')
plt.legend(['PPG signal with highest systolic error', 'PPG signal with highest negative systolic error', 'PPG signal with lowest error (closest to 0)'])


#Bland altman plot map
data1     = np.asarray(y_val[:,2])
data2     = np.asarray(val_predictions[:,2])
mean      = np.mean([data1, data2], axis=0)
diff      = data1 - data2                   # Difference between data1 and data2
md        = np.mean(diff)                   # Mean of the difference
sd        = np.std(diff, axis=0)            # Standard deviation of the difference

print('Mean difference mean arterial blood pressure validation set:', md)
print('Standard deviation of difference mean arterial blood pressure validation set:',sd)
plt.figure()
plt.scatter(mean, diff, c=zmap,s=10)
plt.axhline(md,           color='gray', linestyle='--')
plt.text(130,md+2*sd, '+sd1.96', fontsize=10)
plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
plt.text(130,md, 'Mean diff', fontsize=10)
plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
plt.text(130,md-2*sd, '-sd1.96', fontsize=10)
plt.title('Bland-Altman Plot MAP')
plt.show()

#BHS table
table=[['','<5 mmHg error [%]', '<10 mmHg error [%]', '<15 mmHg error [%]', 'BHS grade','R', 'Mean error','Standard dev'],['DBP',percentage5dbp,percentage10dbp,percentage15dbp,' ' ,Rvaluedbp,meanerrordbp,stdevdbp],['SBP',percentage5sbp,percentage10sbp,percentage15sbp, ' ',Rvaluesbp,meanerrorsbp,stdevsbp ],['MAP',percentage5map,percentage10map,percentage15map,' ', Rvaluemap,meanerrormap,stdevmap]]
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

# %% Test predictions
test_predictions = model.predict(x_test)


scorestrain=model.evaluate(x_train,y_train,verbose=2)
scoresval=model.evaluate(x_val,y_val,verbose=2)
scorestest=model.evaluate(x_test,y_test,verbose=2)


mse_train   = history.history['mse']
mse_val     = history.history['val_mse']
mae_train   = history.history['mae']
mae_val     = history.history['val_mae']

meanerrordbp=statistics.mean(y_test[:,0]-test_predictions[:,0])
stdevdbp=statistics.stdev(y_test[:,0]-test_predictions[:,0])

meanerrorsbp=statistics.mean(y_test[:,1]-test_predictions[:,1])
stdevsbp=statistics.stdev(y_test[:,1]-test_predictions[:,1])

meanerrormap=statistics.mean(y_test[:,2]-test_predictions[:,2])
stdevmap=statistics.stdev(y_test[:,2]-test_predictions[:,2])

print(meanerrordbp,stdevdbp)
print(meanerrorsbp,stdevsbp)
# %% LOSS PLOT
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
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()



totalamount=y_test.shape[0]
# BHS standards DBP
differencesdbp=abs(test_predictions[:,0]-y_test[:,0])
listlower5dbp = [i for i in differencesdbp if i < 5]
listlower10dbp = [i for i in differencesdbp if i < 10]
listlower15dbp = [i for i in differencesdbp if i < 15]
percentage5dbp=(len(listlower5dbp)/totalamount)*100
percentage10dbp=(len(listlower10dbp)/totalamount)*100
percentage15dbp=(len(listlower15dbp)/totalamount)*100




# BHS standards SBP
differencessbp=abs(test_predictions[:,1]-y_test[:,1])
listlower5sbp = [i for i in differencessbp if i < 5]
listlower10sbp = [i for i in differencessbp if i < 10]
listlower15sbp = [i for i in differencessbp if i < 15]
percentage5sbp=(len(listlower5sbp)/totalamount)*100
percentage10sbp=(len(listlower10sbp)/totalamount)*100
percentage15sbp=(len(listlower15sbp)/totalamount)*100



# BHS standards MAP
differencesmap=abs(test_predictions[:,2]-y_test[:,2])
listlower5map = [i for i in differencesmap if i < 5]
listlower10map = [i for i in differencesmap if i < 10]
listlower15map = [i for i in differencesmap if i < 15]
percentage5map=(len(listlower5map)/totalamount)*100
percentage10map=(len(listlower10map)/totalamount)*100
percentage15map=(len(listlower15map)/totalamount)*100

ytestpredictionsdbp = np.vstack([y_test[:,0],test_predictions[:,0]])
zdbp = gaussian_kde(ytestpredictionsdbp)(ytestpredictionsdbp)

plt.figure()
plt.scatter(y_test[:,0], test_predictions[:,0],c=zdbp,s=10) 
plt.plot([40, 140], [40, 140], '--k') 
plt.title('Correlation plot Diastolic Blood Pressure ')
plt.ylabel('Predicted DBP [mmHg]',fontsize=18)
plt.xlabel('Actual DBP [mmHg]',fontsize=18)

# Dit wordt gebruikt voor plotten voor verslag
# plt.figure()
# plt.scatter(y_test[:,0], test_predictions[:,0],c=zdbp,s=10) 
# plt.plot([40, 140], [40, 140], '--k') 
# plt.ylabel('Predicted DBP [mmHg]',fontsize=16)
# plt.xlabel('Actual DBP [mmHg]',fontsize=16)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)


ytestpredictionssbp = np.vstack([y_test[:,1],test_predictions[:,1]])
zsbp = gaussian_kde(ytestpredictionssbp)(ytestpredictionssbp)
plt.figure()
plt.scatter(y_test[:,1], test_predictions[:,1],c=zsbp,s=10) 
plt.plot([80, 200], [80, 200], '--k') 
plt.title('Correlation plot Systolic Blood Pressure')
plt.ylabel('Predicted SBP [mmHg]')
plt.xlabel('Actual SBP [mmHg]')

ytestpredictionsmap = np.vstack([y_test[:,2],test_predictions[:,2]])
zmap = gaussian_kde(ytestpredictionsmap)(ytestpredictionsmap)
plt.figure()
plt.scatter(y_test[:,2], test_predictions[:,2],c=zmap,s=10) 
plt.plot([60, 160], [60, 160], '--k') 
plt.title('Correlation plot Mean Arterial Blood Pressure')
plt.ylabel('Predicted MAP [mmHg]')
plt.xlabel('Actual MAP [mmHg]')



# Correlation coefficient
[Rvaluedbp,pvaluedbp]= pearsonr(y_test[:,0], test_predictions[:,0])

[Rvaluesbp,pvaluesbp]= pearsonr(y_test[:,1], test_predictions[:,1])

[Rvaluemap,pvaluemap]= pearsonr(y_test[:,2], test_predictions[:,2])

print('Pearsons correlation diastolic blood pressure: %.3f' % Rvaluedbp)
print('Pearsons correlation systolic blood pressure: %.3f' % Rvaluesbp)
print('Pearsons correlation mean arterial pressure: %.3f' % Rvaluemap)



#Bland altman plot diastolic blood pressure
data1     = np.asarray(y_test[:,0])
data2     = np.asarray(test_predictions[:,0])
mean      = np.mean([data1, data2], axis=0)
diff      = data1 - data2                   # Difference between data1 and data2
md        = np.mean(diff)                   # Mean of the difference
sd        = np.std(diff, axis=0)            # Standard deviation of the difference

print('Mean difference diastolic blood pressure test set:', md)
print('Standard deviation of difference diastolic blood pressure test set:', sd)
plt.figure()
plt.scatter(mean, diff, c=zdbp,s=10)
plt.axhline(md,           color='gray', linestyle='--')
plt.text(118,md+1.96*sd+0.4, '+sd1.96', fontsize=10)
plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
plt.text(118,md+0.4, 'Mean diff', fontsize=10)
plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
plt.text(118,md-1.96*sd+0.4, '-sd1.96', fontsize=10)
plt.title('Bland-Altman Plot Diastolic Blood Pressure')
plt.xlabel('Diastolic Blood Pressure [mmHg]')
plt.ylabel('Error [mmHg]')
plt.show()


# Find outliers
idxdbp = (np.abs(diff - 0)).argmin()
bestpreddbp=indices_test[idxdbp]
maxerrordbp=np.where(diff==max(diff))[0][0] #Dit is maximale positieve error
minerrordbp=np.where(diff==min(diff))[0][0] #Dit is maximale negatieve error
indexmaxerror=indices_test[maxerrordbp]
indexminerror=indices_test[minerrordbp]
plt.figure()
plt.plot(x_values[indexmaxerror,:])
plt.plot(x_values[indexminerror,:])
plt.plot(x_values[bestpreddbp,:])
plt.title('PPG signals with highest and lowest error diastolic blood pressure')
plt.legend(['PPG signal with highest positive diastolic error', 'PPG signal with highest negative diastolic error', 'PPG signal with lowest error (closest to 0)'])


#Bland altman plot Systolic blood pressure
data1     = np.asarray(y_test[:,1])
data2     = np.asarray(test_predictions[:,1])
mean      = np.mean([data1, data2], axis=0)
diff      = data1 - data2                   # Difference between data1 and data2
md        = np.mean(diff)                   # Mean of the difference
sd        = np.std(diff, axis=0)            # Standard deviation of the difference

print('Mean difference systolic blood pressure test set:', md)
print('Standard deviation of difference systolic blood pressure test set:', sd)
plt.figure()
plt.scatter(mean, diff, c=zsbp,s=10)
plt.axhline(md,           color='gray', linestyle='--')
plt.text(68,md+1.96*sd+0.4, '+sd1.96', fontsize=10)
plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
plt.text(68,md+0.4, 'Mean diff', fontsize=10)
plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
plt.text(68,md-1.96*sd+0.4, '-sd1.96', fontsize=10)
plt.title('Bland-Altman Plot Systolic Blood Pressure')
plt.xlabel('Systolic Blood Pressure [mmHg]')
plt.ylabel('Error [mmHg]')
plt.show()

# %% Find outliers
idxsbp = (np.abs(diff - 0)).argmin()
bestpredsbp=indices_test[idxsbp]
maxerrorsbp=np.where(diff==max(diff))[0][0]
minerrorsbp=np.where(diff==min(diff))[0][0]
indexmaxerror=indices_test[maxerrorsbp]
indexminerror=indices_test[minerrorsbp]
plt.figure()
plt.plot(x_values[indexmaxerror,:])
plt.plot(x_values[indexminerror,:])
plt.plot(x_values[bestpredsbp,:])
plt.title('PPG signals with highest and lowest error systolic blood pressure')
plt.legend(['PPG signal with highest systolic error', 'PPG signal with highest negative systolic error', 'PPG signal with lowest error (closest to 0)'])


#Bland altman plot map
data1     = np.asarray(y_test[:,2])
data2     = np.asarray(test_predictions[:,2])
mean      = np.mean([data1, data2], axis=0)
diff      = data1 - data2                   # Difference between data1 and data2
md        = np.mean(diff)                   # Mean of the difference
sd        = np.std(diff, axis=0)            # Standard deviation of the difference

print('Mean difference mean arterial blood pressure validation set:', md)
print('Standard deviation of difference mean arterial blood pressure validation set:',sd)
plt.figure()
plt.scatter(mean, diff, c=zmap,s=10)
plt.axhline(md,           color='gray', linestyle='--')
plt.text(135,md+1.96*sd+0.4, '+sd1.96', fontsize=10)
plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
plt.text(135,md+0.4, 'Mean diff', fontsize=10)
plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
plt.text(135,md-1.96*sd+0.4, '-sd1.96', fontsize=10)
plt.title('Bland-Altman Plot Mean Arterial Pressure')
plt.xlabel('Mean Arterial Pressure [mmHg]')
plt.ylabel('Error [mmHg]')
plt.show()

# #Bland altman plot Systolic blood pressure zoals in verslag
# data1     = np.asarray(y_test[:,1])
# data2     = np.asarray(test_predictions[:,1])
# mean      = np.mean([data1, data2], axis=0)
# diff      = data1 - data2                   # Difference between data1 and data2
# md        = np.mean(diff)                   # Mean of the difference
# sd        = np.std(diff, axis=0)            # Standard deviation of the difference

# print('Mean difference systolic blood pressure test set:', md)
# print('Standard deviation of difference systolic blood pressure test set:', sd)
# plt.figure()
# plt.scatter(mean, diff, c=zsbp,s=10)
# plt.axhline(md,           color='gray', linestyle='--')
# plt.text(68,md+1.96*sd+0.4, '+sd1.96', fontsize=12)
# plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
# plt.text(68,md+0.4, 'Mean diff', fontsize=12)
# plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
# plt.text(68,md-1.96*sd+0.4, '-sd1.96', fontsize=12)
# plt.xlabel('Systolic Blood Pressure [mmHg]',fontsize=16)
# plt.ylabel('Error [mmHg]',fontsize=16)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.show()

# %% Find outliers
idxmap = (np.abs(diff - 0)).argmin()
bestpredmap=indices_test[idxmap]
maxerrormap=np.where(diff==max(diff))[0][0]
minerrormap=np.where(diff==min(diff))[0][0]
indexmaxerror=indices_test[maxerrormap]
indexminerror=indices_test[minerrormap]
plt.figure()
plt.plot(x_values[indexmaxerror,:])
plt.plot(x_values[indexminerror,:])
plt.plot(x_values[bestpredsbp,:])
plt.title('PPG signals with highest and lowest error mean arterial pressure')
plt.legend(['PPG signal with highest mean arterial pressure error', 'PPG signal with highest negative mean arterial pressure error', 'PPG signal with lowest error (closest to 0)'])

#BHS table
table=[['','<5 mmHg error [%]', '<10 mmHg error [%]', '<15 mmHg error [%]', 'BHS grade','R', 'Mean error','Standard dev'],['DBP',percentage5dbp,percentage10dbp,percentage15dbp,' ' ,Rvaluedbp,meanerrordbp,stdevdbp],['SBP',percentage5sbp,percentage10sbp,percentage15sbp, ' ',Rvaluesbp,meanerrorsbp,stdevsbp ],['MAP',percentage5map,percentage10map,percentage15map,' ', Rvaluemap,meanerrormap,stdevmap]]
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

q25dbp, q75dbp = np.percentile(differencesdbp, [25, 75])
bin_widthdbp = 2 * (q75dbp - q25dbp) * len(differencesdbp) ** (-1/3)
binsdbp = round((differencesdbp.max() - differencesdbp.min()) / bin_widthdbp)

#BHS Histogram plot dbp
plt.figure()
plt.style.use('ggplot')
plt.hist(differencesdbp,bins=binsdbp,color='b')
plt.axvline(5, color='k', linestyle='dashed', linewidth=1)
plt.axvline(10, color='k', linestyle='dashed', linewidth=1)
plt.axvline(15, color='k', linestyle='dashed', linewidth=1)
plt.title('BHS grading histogram Diastolic Blood Pressure')
plt.xlabel('Error [mmHg]')
plt.ylabel('Count')
plt.show()


q25sbp, q75sbp = np.percentile(differencessbp, [25, 75])
bin_widthsbp = 2 * (q75sbp - q25sbp) * len(differencessbp) ** (-1/3)
binssbp = round((differencessbp.max() - differencessbp.min()) / bin_widthsbp)

#BHS Histogram plot sbp
plt.figure()
plt.style.use('ggplot')
plt.hist(differencessbp,bins=binssbp,color='b')
plt.axvline(5, color='k', linestyle='dashed', linewidth=1)
plt.axvline(10, color='k', linestyle='dashed', linewidth=1)
plt.axvline(15, color='k', linestyle='dashed', linewidth=1)
plt.title('BHS grading histogram Systolic Blood Pressure')
plt.xlabel('Error [mmHg]')
plt.ylabel('Count')
plt.show()


q25map, q75map = np.percentile(differencesmap, [25, 75])
bin_widthmap = 2 * (q75map - q25map) * len(differencesmap) ** (-1/3)
binsmap = round((differencesmap.max() - differencesmap.min()) / bin_widthmap)

#BHS Histogram plot map
plt.figure()
plt.style.use('ggplot')
plt.hist(differencesmap,bins=binsmap,color='b')
plt.axvline(5, color='k', linestyle='dashed', linewidth=1)
plt.axvline(10, color='k', linestyle='dashed', linewidth=1)
plt.axvline(15, color='k', linestyle='dashed', linewidth=1)
plt.title('BHS grading histogram Mean Arterial Pressure')
plt.xlabel('Error [mmHg]')
plt.ylabel('Count')
plt.show()



# %% FEATURE MAPS

# retrieve weights from the second hidden layer, hier kan je alle conv layers kiezen. Hangt van de layer af hoeveel channels het heeft, alles heeft er 3 behalve de eerste layer.
filters, biases = model.layers[1].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = plt.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(np.expand_dims(f[:,j],axis=1), cmap='gray')
		ix += 1
# show the figure
plt.show()


# redefine model to output right after the first hidden layer
ixs = [1,4,7,10]
outputs = [model.layers[i].output for i in ixs]
newmodel = Model(inputs=model.inputs, outputs=outputs)
x_test_example1=np.expand_dims(x_test[0,:],axis=0)
feature_maps=newmodel.predict(x_test_example1)

squarerow=6
squarecolumn=8
fmap=feature_maps[0]
lengtey=np.arange(1,fmap.shape[1]+1)
xsignal=np.interp(np.arange(0, test.shape[1], 1), np.arange(0, test.shape[1]), test[0,:,0])
for fmap in feature_maps:
	plt.figure()
	# plot all 64 maps in an 8x8 squares
	ix = 1
	for _ in range(squarecolumn):
		for _ in range(squarerow):
			# specify subplot and turn of axis
			ax = plt.subplot(squarerow, squarecolumn, ix,  fc='black')
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			plt.scatter(lengtey,xsignal,c=fmap[0,:,ix-1],cmap='gray',marker=".",s=10)
			ix += 1
	# show the figure
	plt.show()



print('End of CNN analysis')

#per convolutional layer apart werkt dat zo:
squarerow=3
squarecolumn=8
fmap=feature_maps[3]
lengtey=np.arange(1,fmap.shape[1]+1)
xsignal=np.interp(np.arange(0, test.shape[1], 8), np.arange(0, test.shape[1]), test[0,:,0])
plt.figure()
# plot all 64 maps in an 8x8 squares
ix = 1
for _ in range(squarecolumn):
	for _ in range(squarerow):
		# specify subplot and turn of axis
		ax = plt.subplot(squarerow, squarecolumn, ix,  fc='black')
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.scatter(lengtey,xsignal,c=fmap[0,:,ix-1],cmap='gray',marker=".",s=10)
		ix += 1
# show the figure
plt.suptitle('Feature maps test example 1 convolutional layer 8')
plt.show()


# %%
