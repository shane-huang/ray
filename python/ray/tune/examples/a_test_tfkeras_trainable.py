import ray
from ray.tune import register_trainable, grid_search, run_experiments
from ray.tune import sample_from
from ray.tune import Trainable

from ray.tune.util import pin_in_object_store, get_pinned_object
from ray.tune.schedulers import HyperBandScheduler, AsyncHyperBandScheduler
from ray.tune.suggest import BayesOptSearch, HyperOptSearch

import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

import tensorflow as tf



ray.init()

def load_data():
    data = pd.read_csv('./O3.csv')
    train, test = train_test_split(data, test_size=0.1)
    data_id = pin_in_object_store(data)
    train_id= pin_in_object_store(train)
    test_id = pin_in_object_store(test)
    return data_id,train_id,test_id

#load data and pin to object store
data_id, train_id, test_id = load_data()

def evaluate(y_true,y_pred):
    mse = metrics.mean_squared_error(y_true,y_pred)
    r2=metrics.r2_score(y_true,y_pred)
    return (mse,r2)

class TuneCallback(tf.keras.callbacks.Callback):
    def __init__(self, reporter, logs={}):
        self.reporter = reporter
        self.iteration = 0


    def on_train_end(self, epoch, logs={}):
        self.reporter(
            timesteps_total=self.iteration,
            done=1,
            epochs_total=epoch,
            neg_mse=(-1)*logs['mean_squared_error']
        )

    def on_batch_end(self, batch, logs={}):
        self.iteration += 1
        self.reporter(
            timesteps_total=self.iteration,
            neg_mse=(-1)*logs['mean_squared_error']
        )

class MyTrainable(Trainable):

    def _prepare_data(self):
        self.trainset = get_pinned_object(train_id)
        self.testset = get_pinned_object(test_id)
        self.x_train = self.trainset[['HUMIDITY', 'TEMPERATURE', 'O3']].values
        self.y_train = self.trainset['O3_LABEL'].values
        self.x_test = self.testset[['HUMIDITY', 'TEMPERATURE', 'O3']].values
        self.y_test = self.testset['O3_LABEL'].values
        #return (x_train, y_train), (x_test, y_test)

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(int(self.config['hidden_layers1'])))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(int(self.config['hidden_layers2'])))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.config['lr']),
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        return model

    def _setup(self,config):
        self.config = config
        #self.fullset= get_pinned_object(data_id),
        self._prepare_data()
        self.model = self._build_model()

    def _train(self):
        # We set threads here to avoid contention, as Keras
        # is heavily parallelized across multiple cores.
        #self.model.fit(self.x_train, self.y_train, epochs=self.config['epochs'],callbacks=[TuneCallback(reporter)])
        self.model.fit(self.x_train, self.y_train, epochs=self.config['epochs'])
        y_predicted = self.model.predict(self.x_test)
        test_mse, test_r2 = evaluate(self.y_test, y_predicted)
        return {"neg_mse": (-1)*test_mse, "r2":test_r2}


    def _save(self, checkpoint_dir):
        file_path = checkpoint_dir + "/model"
        self.model.save_weights(file_path)
        return file_path

    def _restore(self, path):
        self.model.load_weights(path)

    def _stop(self):
        pass

#register_trainable("train_nn", train_nn)
register_trainable("trainable_nn", MyTrainable)

from hyperopt import hp


trail_scheduler = AsyncHyperBandScheduler(
    reward_attr="neg_mse"
)


print("***************************** trying 2 hidden layers ***************************")
space_2hidden = {
         "lr": (0.001, 0.1),
         "hidden_layers1": (1, 100),
         "hidden_layers2": (1, 100),
    }
opt_search_2hidden = BayesOptSearch(
    space_2hidden,
    max_concurrent=4,
    reward_attr="neg_mse",
    utility_kwargs={
        "kind": "ucb",
        "kappa": 2.5,
        "xi": 0.0
    })

#run_experiments(config_2hidden, search_alg=opt_search_2hidden, scheduler=trail_scheduler)

## Trainable style
config_2hidden_trainable = {
     "2hidden_search_trainable": {
        "run": "trainable_nn",
        "stop": { "neg_mse": -700 },
        #"resources_per_trial": {
        #    "cpu": 4
        #},
        "num_samples": 10,
        "config": {
            "epochs" : 500,
            "hidden_num": 2,
            #"iterations": 100,
            #"max_iter": 10000,
        },
    }
}

run_experiments(config_2hidden_trainable,search_alg=opt_search_2hidden, scheduler=trail_scheduler)