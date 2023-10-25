#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

# Related to plot parameters
plt.rcParams['figure.dpi']  = 250
plt.rcParams['savefig.dpi'] = 250








# Read the csv files
def read_tree(tree_path):
    
    df = pd.read_csv(tree_path)
    
    if tree_path == "simple_shower_nn.csv":
        y = 0
    
    elif tree_path == "vincia_nn.csv":
        y = 1
        
    elif tree_path == "dire_nn.csv":
        y = 2
        
    else:
        raise NameError('Wrongly named csv')
    
    df ['y'] = y
        
    
        
    return df
    
    

# Read the data
vincia = read_tree("vincia_nn.csv")
simple = read_tree("simple_shower_nn.csv")

print (f"simple = {simple}")
print (f"vincia = {vincia}")



print ('% of simple shower events	= ', len (simple)/ ( len(simple) + len(vincia) ) * 100, '%' )
print ('% of vincia events		= ', len (vincia)/ ( len(simple) + len(vincia) ) * 100, '%' )
print ('\n')



# Prepare X and y
X = pd.concat([vincia,simple])





# Split into training and test sets
from sklearn.model_selection import train_test_split

X_train_full, X_test = train_test_split(X, test_size = 0.25)
X_train, X_valid     = train_test_split(X_train_full, test_size = 0.3)


print(f'X_train = {X_train}')

y	= X      ['y'].to_numpy()
y_train = X_train['y'].to_numpy()
y_valid = X_valid['y'].to_numpy()
y_test  = X_test ['y'].to_numpy()

X       = X.drop	    (['y'],axis=1)
X_train = X_train.drop(['y'],axis=1)
X_valid = X_valid.drop(['y'],axis=1)
X_test  = X_test.drop (['y'],axis=1)

X       = X.to_numpy()
X_train = X_train.to_numpy()
X_valid = X_valid.to_numpy()
X_test  = X_test.to_numpy()

print(f'X_train (as numpy)= \n{X_train}')
#Let's write these into a csv
import csv

# This prepares all sets to be prepared as a list and then to csv.
def set_to_csv (X,y):
    X_simple = []
    X_vincia = []
    
    for i in range (len (X)):
        if y[i] == 1:
            X_vincia.append (X [i] )
        
        else:
            X_simple.append (X [i] )
    
    X_simple = np.array (X_simple)
    X_vincia = np.array (X_vincia)
        
    return {'simple': X_simple, 'vincia': X_vincia}
    
full_simple  = set_to_csv(X,y)['simple']
full_vincia  = set_to_csv(X,y)['vincia']
	
train_simple = set_to_csv(X_train, y_train)['simple']
train_vincia = set_to_csv(X_train, y_train)['vincia']

valid_simple = set_to_csv(X_valid, y_valid)['simple']
valid_vincia = set_to_csv(X_valid, y_valid)['vincia']

test_simple  = set_to_csv(X_test,  y_test)['simple']
test_vincia  = set_to_csv(X_test,  y_test)['vincia']

print (f"train_simple.shape = {train_simple.shape}")

#Get the headers from the original csv, in order to keep the same headers		
header = list (pd.read_csv("simple_shower_nn.csv").columns) 
header = ",".join(header)
print (f'header = {header}')


print ("Writing train, valid, and test sets to csv...\n")
np.savetxt('csv/full_vincia.csv',  full_vincia,   delimiter=',', header = header, comments="")
np.savetxt('csv/full_simple.csv',  full_simple,   delimiter=',', header = header, comments="")
np.savetxt('csv/train_vincia.csv', train_vincia,  delimiter=',', header = header, comments="")
np.savetxt('csv/train_simple.csv', train_simple,  delimiter=',', header = header, comments="")
np.savetxt('csv/valid_vincia.csv', valid_vincia,  delimiter=',', header = header, comments="")
np.savetxt('csv/valid_simple.csv', valid_simple,  delimiter=',', header = header, comments="")
np.savetxt('csv/test_vincia.csv' , test_vincia ,  delimiter=',', header = header, comments="")
np.savetxt('csv/test_simple.csv' , test_simple ,  delimiter=',', header = header, comments="")
print ("Written!")


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train.astype(np.float64))
X_valid = scaler.transform(X_valid.astype(np.float64))
X_test  = scaler.transform(X_test.astype(np.float64))













# In[23]:
size = X_train.shape[1:]

# Make a model. The input is a set of sequences with len(size) length
# predicting with sort of model to implement
input = keras.layers.Input(shape=size)
hidden1 = keras.layers.Dense(69, activation="relu")(input)
hidden2 = keras.layers.Dense(112, activation="relu")(hidden1)
output  = keras.layers.Dense(1)(hidden2)


model = keras.models.Model(inputs=[input], outputs=[output])

# Determien the optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.01)

print ('\n')


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
optimizer="adam",
metrics=["accuracy"] )

print ("Fitting model")

# Fit the model
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid),
                    callbacks=[keras.callbacks.EarlyStopping(patience=10)] )






# Get the model's summary and weights                    
print (model.summary() )


weights1, biases1 = model.layers[1].get_weights()
weights2, biases2 = model.layers[2].get_weights()
weights3, biases3 = model.layers[3].get_weights()


"""
with np.printoptions(threshold=np.inf):

	print ('Hidden layer 1 weights: \n', weights1 )
	print ('Hidden layer 1 biases : \n', biases1  )
	print ('\n\n\n\n')

	print ('Hidden layer 2 weights: \n', weights2 )
	print ('Hidden layer 2 biases : \n', biases2  )
	print ('\n\n\n\n')

	print ('Hidden layer 3 weights: \n', weights3 )
	print ('Hidden layer 3 biases : \n', biases3  )
	print ('\n\n\n\n')


"""





# Evaluate the model on the various sets

print("\n\nEvaluating the model...\n")

print('Training set:')
train_score, train_acc = model.evaluate(X_train, y_train)

print('\nValidation set:')
valid_score, valid_acc = model.evaluate(X_valid, y_valid)

print('\nTest set:')
test_score,  test_acc  = model.evaluate(X_test , y_test)   

print('\nModel evaluated!')
from sklearn.metrics import precision_score, recall_score

# This model.predict(X_test) would give us a logit result. To translate it into probabilities, we must use the transformation
# Probability = 1 / (1 + exp(- logit ) )
print("\n\nMaking the predictions...\n")

y_train_pred = 1/(1+ np.exp(-model.predict(X_train)) ) 
y_valid_pred = 1/(1+ np.exp(-model.predict(X_valid)) )
y_pred       = 1/(1+ np.exp(-model.predict(X_test )) )

y_train_pred_binary	= np.where(y_train_pred <= 0.5, 0, 1)
y_valid_pred_binary	= np.where(y_valid_pred <= 0.5, 0, 1)
y_pred_binary		= np.where(y_pred       <= 0.5, 0, 1)


# Arrays to be created for the distributions of specific y values
y_pred0 = [] 
y_pred1 = [] 

for i in range (len (y_test)) :
	if y_test[i] == 0:
		y_pred0.append (y_pred[i])
	else:
		y_pred1.append (y_pred[i])
		
y_pred0 = np.asarray (y_pred0)
y_pred1 = np.asarray (y_pred1)

print ("\nSets predicted!")











# ROC curve and scores

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


print ("\n\nCalculating ROC curves...")
train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_pred)
print ('Area under ROC curve (training set): ', train_auc)

valid_fpr, valid_tpr, valid_thresholds = roc_curve(y_valid, y_valid_pred)
valid_auc = roc_auc_score(y_valid, y_valid_pred)
print ('Area under ROC curve (validation set): ', valid_auc)

test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_pred)
print ('Area under ROC curve (test set): ', test_auc)


# Locate the ideal threshold for the curve
# calculate the Youden’s J statistic
J_train = train_tpr - train_fpr
train_ix = np.argmax(J_train)
best_thresh_train = train_thresholds[train_ix]

J_valid = valid_tpr - valid_fpr
valid_ix = np.argmax(J_valid)
best_thresh_valid = valid_thresholds[valid_ix]

J_test = test_tpr - test_fpr
test_ix = np.argmax(J_test)
best_thresh_test = test_thresholds[test_ix]

print('\n\n')
print('Best Threshold (train) = %f' % (best_thresh_train))
print('Best Threshold (valid) = %f' % (best_thresh_valid))
print('Best Threshold (test ) = %f' % (best_thresh_test))





print ("\nROC curve drawn!")

print ("\nCalculating scores...")

print ("Train set score:     ", train_score)
print ("Train set accuracy:  ", train_acc  )
print ("Train set precision: ", precision_score(y_train_pred_binary, y_train) )
print ("Train set recall:    ", recall_score   (y_train_pred_binary, y_train) )

print ("\n")

print ("Validation set score:     ", valid_score)
print ("Validation set accuracy:  ", valid_acc  )
print ("Validation set precision: ", precision_score(y_valid_pred_binary, y_valid) )
print ("Validation set recall:    ", recall_score   (y_valid_pred_binary, y_valid) )
                
                 
print ("\n")

print ("Test set score:     ", test_score)
print ("Test set accuracy:  ", test_acc  )   
print ("Test set precision: ", precision_score(y_pred_binary, y_test) )
print ("Test set recall:    ", recall_score   (y_pred_binary, y_test) )                 
                    
print ("\nScores calculated!")    
                   
 
 
print ("\n\n") 
                    
                    
                    








# In[196]:

# Plot the loss & accuracy curve

import pandas as pd
fig = plt.figure()
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.xlabel('Epoch')
plt.savefig('plots/loss&accuracy_3layers.png')
plt.show()




# Plot the ROC curve
def plot_roc_curve(fpr, tpr, ix, name = ''):
	plt.plot(fpr, tpr, linewidth=1, label=name)
	plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate' )
	plt.title( 'ROC Curve')
	# Set the point with the greatest threshold
	plt.scatter(fpr[ix], tpr[ix], marker='o', label='Best')
	




# Plot the predictions
	

fig = plt.figure()
plt.hist(y_pred0, density = True, bins = 100, histtype='step', label = 'Simple shower')
plt.hist(y_pred1, density = True, bins = 100, histtype='step', label = 'Vincia')
plt.title('Distribution of the predictions for $X_{test}$, with vincia sample')
plt.xlabel('$y_{pred}$')
plt.legend(loc="best")
plt.savefig('plots/y_pred.png')
plt.show()





X_test_confident = []
y_test_confident = []



# Create csv's with the data for confident and doubtful results.
# Here, we assume that we're confident that X_test predictions under 0.3 will be accurate.

print ("Building the confident, doubtful, and predicted sets...")
import csv

confident_vincia = []
confident_simple = []

doubtful_vincia  = []
doubtful_simple  = []


for i in range (len (y_pred) ):
	if (y_pred[i]< 0.3) or (y_pred[i] > 0.6):
		X_test_confident.append(X_test[i] )
		y_test_confident.append(y_test[i] )
		
		
		
		if y_test[i] == 0:
			confident_simple.append( X_test[i] )
		
		elif y_test[i] == 1:
			confident_vincia.append( X_test[i] )
		
	else:
		if y_test[i] == 0:
			doubtful_simple.append( X_test[i] )
		
		elif y_test[i] == 1:
			doubtful_vincia.append( X_test[i] )


#y_confident_pred 	  = 1/(1+ np.exp(-model.predict(X_test_confident) ) )
#y_confident_pred_binary   = np.where(y_pred_confident <= 0.5, 0, 1)
#print ('y_confident_pred calculated')

confident_vincia = scaler.inverse_transform( np.asarray (confident_vincia) ) # We have to invert the Standard Scaler, and convert the list into an np array
confident_simple = scaler.inverse_transform( np.asarray (confident_simple) )
doubtful_vincia  = scaler.inverse_transform( np.asarray (doubtful_vincia)  )
doubtful_simple  = scaler.inverse_transform( np.asarray (doubtful_simple)  )
print('Confident vincia inverse transformed')

np.savetxt('csv/confident_vincia.csv', confident_vincia,  delimiter=',', header = header, comments="")
np.savetxt('csv/confident_simple.csv', confident_simple,  delimiter=',', header = header, comments="")
np.savetxt('csv/doubtful_vincia.csv',  doubtful_vincia  , delimiter=',', header = header, comments="")
np.savetxt('csv/doubtful_simple.csv',  doubtful_simple  , delimiter=',', header = header, comments="")
			
X_test_confident = np.asarray (X_test_confident)
y_test_confident = np.asarray (y_test_confident)


		

# Create csv's with the data for the output results.

predicted_vincia = []
predicted_simple = []





for i in range (len (y_pred) ):
	if (y_pred[i]< 0.5):
		predicted_simple.append(X_test[i] )
		
	else:
		predicted_vincia.append(X_test[i] )
		
	



predicted_vincia  = scaler.inverse_transform( np.asarray (predicted_vincia)  )
predicted_simple  = scaler.inverse_transform( np.asarray (predicted_simple)  )


np.savetxt('csv/predicted_vincia.csv',  predicted_vincia  , delimiter=',', header = header, comments="")
np.savetxt('csv/predicted_simple.csv',  predicted_simple , delimiter=',', header = header, comments="")

print ("Sets written to csv!")

print ("""\n\n\nConfident results: we select only events whose probability is less than 10% from 0 (simple shower) or 1 (vincia shower).
With this, we can take an arbitrary shower, take only the most certain inputs, and with those classify the shower as resembling one of those models.\n""")



# Predictions vs labels. Default label at 0.5 and label at the ideal threshold.

import sys
np.set_printoptions(suppress=True) # Disable scientific notation
np.set_printoptions(threshold=sys.maxsize) # Show the entire numpy array

print ('Predicted probability vs. actual result for 100 events: ')
for i in range (30):
	print (y_pred[i], y_test[i] )
	
y_pred_short = y_pred[:10000]
y_test_short = y_test[:10000]
	

m_default = tf.keras.metrics.binary_accuracy(y_pred_short, y_test_short, threshold=0.5)
print("With threshold at 0.5, accuracy is:", m_default[0])

m_best = tf.keras.metrics.binary_accuracy(y_pred_short, y_test_short, threshold = best_thresh_test)
print(f"With threshold at {best_thresh_test}, test accuracy is: {m_best[0]}")


	
"""
print ('\n\n')
	
print('len(X)       = ', len(X)	)
print('len(X_train) = ', len (X_train)	)
print('len(X_valid) = ', len (X_valid)	)
print('len(X_test)  = ' , len (X_test)	)
print('Len of X_test_confident: ', len(X_test_confident) )
 
print ("Fraction of events kept: ", len(X_test_confident)/len(X_test), "\n\n" )


"""
print ("\n\n")

#confident_fpr, confident_tpr, confident_thresholds = roc_curve(y_confident_pred_binary, y_test_confident)
#confident_auc = roc_auc_score(y_confident_pred_binary, y_test_confident)
#print ('Area under ROC curve (confident events): ', confident_auc, '\n')






score, acc = model.evaluate(X_test_confident, y_test_confident)
print ('Confident events score:		', score)
print ('Confident events accuracy:	', acc)
#print ('Confident events precision:	', precision_score(y_confident_pred_binary, y_test_confident) )
#print ('Confident events recall:	', recall_score   (y_confident_pred_binary, y_test_confident) )


plt.figure()
plot_roc_curve(train_fpr, train_tpr, train_ix, name = 'train')
plot_roc_curve(valid_fpr, valid_tpr, valid_ix, name = 'validation')
plot_roc_curve(test_fpr , test_tpr , test_ix , name = 'test')
#plot_roc_curve(confident_fpr , confident_tpr , name = 'confident')
plt.axhline(recall_score   (y_pred_binary, y_test), label = 'Threshold at 0.5' )
plt.legend()
plt.savefig('plots/ROC.png')
plt.show()

threshold = []
accuracy = []

"""counter = 0
for p in np.unique(model.predict(X_train)):
	#threshold.append(p)
	#y_pred = (model.predict(X_train) >= p).astype(int)
	#accuracy.append(balanced_accuracy_score(y_train,y_pred))
	counter += 1
	if counter % 100 == 0: print (f'count = {counter}')
"""

#plt.scatter(threshold,accuracy)
#plt.xlabel("Threshold")
#plt.ylabel("Balanced accuracy")
#plt.savefig('plots/threshold_vs_accuracy.png')
#plt.show()

#best_thresh = threshold[np.argmax(accuracy)]
#print (f'Recalculated best threshold: {best_thresh}')

#best_accuracy = np.argmax(accuracy)
#print (f'Recalculated best accuracy:  {best_accuracy}')

print ('\nDone!')






