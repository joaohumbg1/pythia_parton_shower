#!/usr/bin/env python
# coding: utf-8

# # Import packages

# In[53]:


"""
Optuna example that demonstrates a pruner for Tensorflow (Estimator API).
In this example, we optimize the hyperparameters of a neural network for hand-written
digit recognition in terms of validation accuracy. The network is implemented by Tensorflow and
evaluated by MNIST dataset. Throughout the training of neural networks, a pruner observes
intermediate results and stops unpromising trials.
You can run this example as follows:
    $ python tensorflow_estimator_integration.py
"""

import shutil
import tempfile
import urllib

import optuna
from optuna.trial import TrialState
import tensorflow_datasets as tfds

import tensorflow as tf
import pandas as pd
import numpy as np


# # Prepare the data

# In[54]:


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
        
    return (df,y)

def read_Xy (tree_path):
    
    tree = read_tree (tree_path)
    X = tree[0].to_numpy()
    y = tree[1] * np.ones (len(tree[0]))
    
    return {'X': X, 'y': y}


# In[55]:


# Read the data
vincia_data = read_Xy("vincia_nn.csv")
simple_data = read_Xy("simple_shower_nn.csv")


# In[56]:





# In[57]:


# Prepare X and y
X = np.concatenate( (vincia_data['X'],simple_data['X']) )
y = np.concatenate( (vincia_data['y'],simple_data['y']) )
y = y.astype(int)

def full_data(X,y):
    return {'X': X, 'y':y}


# In[58]:


# Split into training and test sets
from sklearn.model_selection import train_test_split

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size = 0.2)
X_train, X_valid, y_train, y_valid  = train_test_split(X_train_full, y_train_full, test_size = 0.2)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train.astype(np.float64))
X_valid = scaler.transform(X_valid.astype(np.float64))
X_test  = scaler.transform(X_test.astype(np.float64))


# In[59]:


train_dataset = tf.data.Dataset.from_tensor_slices((X_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test))


# # Optuna preparation

# In[63]:


MODEL_DIR = tempfile.mkdtemp()
BATCH_SIZE = 128
TRAIN_STEPS = 1000
PRUNING_INTERVAL_STEPS = 50
N_TRAIN_BATCHES = 3000
N_VALID_BATCHES = 1000


# In[64]:


def preprocess(data, label):
    data = tf.reshape(data, [-1, len(X[0])])
    return {"x": data}, label


# In[65]:


def train_input_fn():
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.map(preprocess).shuffle(60000).batch(BATCH_SIZE).take(N_TRAIN_BATCHES)
    return train_ds


# In[66]:


def eval_input_fn():
    valid_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    valid_ds = valid_ds.map(preprocess).shuffle(10000).batch(BATCH_SIZE).take(N_VALID_BATCHES)
    return valid_ds


# In[68]:
size = X_train.shape[1:]

def create_classifier(trial):
    # We optimize the numbers of layers and their units.

    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_units = []
    for i in range(n_layers):
        n_units = trial.suggest_int("n_units_l{}".format(i), 1, 128)
        hidden_units.append(n_units)

    config = tf.estimator.RunConfig(
        save_summary_steps=PRUNING_INTERVAL_STEPS, save_checkpoints_steps=PRUNING_INTERVAL_STEPS
    )

    model_dir = "{}/{}".format(MODEL_DIR, trial.number)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=[tf.feature_column.numeric_column("x", shape=size)],
        hidden_units=hidden_units,
        model_dir=model_dir,
        n_classes=10,
        optimizer=lambda: tf.keras.optimizers.Adam(learning_rate=0.01),
        config=config,
    )

    return classifier


# In[69]:


def objective(trial):
    classifier = create_classifier(trial)

    optuna_pruning_hook = optuna.integration.TensorFlowPruningHook(
        trial=trial,
        estimator=classifier,
        metric="accuracy",
        run_every_steps=PRUNING_INTERVAL_STEPS,
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=TRAIN_STEPS, hooks=[optuna_pruning_hook]
    )

    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, start_delay_secs=0, throttle_secs=0)

    eval_results, _ = tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    return float(eval_results["accuracy"])


# # Optimization

# In[70]:


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


    print ('% of simple shower events: ', len (simple_data['X'])/ ( len(simple_data['X']) + len(vincia_data['X']) ) * 100, '%' )
    
    shutil.rmtree(MODEL_DIR)


# In[ ]:
print

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




