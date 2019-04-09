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


def train_nn(config, reporter):
    def prepare_data(train_id,test_id):
        trainset = get_pinned_object(train_id)
        testset = get_pinned_object(test_id)
        x_train = trainset[['HUMIDITY', 'TEMPERATURE', 'O3']].values
        y_train = trainset['O3_LABEL'].values
        x_test = testset[['HUMIDITY', 'TEMPERATURE', 'O3']].values
        y_test = testset['O3_LABEL'].values
        return (x_train, y_train), (x_test, y_test)

    def build_model(config):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(int(config['hidden_layers1'])))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(int(config['hidden_layers2'])))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=config['lr']),
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        return model

    (x_train, y_train), (x_test, y_test) = prepare_data(train_id,test_id)
    model = build_model(config)
    model.fit(x_train, y_train, epochs=config['epochs'], callbacks=[TuneCallback(reporter)])
    #y_predicted = model.predict(x_test)
    #test_mse, test_r2 = evaluate(y_test, y_predicted)

    #reporter(
    #    neg_mse=(-1)*test_mse,
    #    r_square=test_r2,
    #)



    #fullset,trainset,testset=get_data()
    #test_mse, test_r2 = train(trainset,testset,fullset)
    #test_mse, test_r2 = train_tfkeras(trainset,testset,fullset)
    #for i in range(config["iterations"]):

    #reporter(
        #timesteps_total=i,
    #    neg_mse=(-1)*test_mse,
    #    r_square=test_r2,
    #)



register_trainable("train_nn", train_nn)

from hyperopt import hp




trail_scheduler = AsyncHyperBandScheduler(
    reward_attr="neg_mse"
)


config_3hidden = {
     "3hidden_search": {
        "run": "train_nn",
        "stop": { "r_square": 0.75 },
        #"resources_per_trial": {
        #    "cpu": 4
        #},
        "num_samples": 1000,
        "config": {
            "hidden_num": 3,
            "iterations": 100,
            "max_iter": 10000,
        },
    }
}
config_2hidden = {
     "2hidden_search_tf": {
        "run": "train_nn",
        "stop": { "neg_mse": -700 },
        #"resources_per_trial": {
        #    "cpu": 4
        #},
        "num_samples": 10,
        "config": {
            "epochs" : 500,
            "hidden_num": 2,
            "iterations": 100,
            "max_iter": 10000,
        },
    }
}
config_1hidden = {
     "1hidden_search": {
        "run": "train_nn",
        "stop": { "neg_mse": -700 },
        #"resources_per_trial": {
        #    "cpu": 4
        #},
        "num_samples": 1000,
        "config": {
            "hidden_num": 1,
            "iterations": 100,
            "max_iter": 10000,
        },
    }
}

print("***************************** trying 3 hidden layers ***************************")
space_3hidden = {
         "lr": (0.001, 0.1),
         "hidden_layers1": (1, 100),
         "hidden_layers2": (1, 100),
         "hidden_layers3": (1, 100),
    }
opt_search_3hidden = BayesOptSearch(
    space_3hidden,
    max_concurrent=4,
    reward_attr="neg_mse",
    utility_kwargs={
        "kind": "ucb",
        "kappa": 2.5,
        "xi": 0.0
    }
)
#run_experiments(config_3hidden, search_alg=opt_search_3hidden, scheduler=trail_scheduler)

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

run_experiments(config_2hidden, search_alg=opt_search_2hidden, scheduler=trail_scheduler)
