#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random

import numpy as np

import ray
from ray.tune import Trainable, run_experiments, Experiment, sample_from
from ray.tune.schedulers import HyperBandScheduler

from ray.tune.util import pin_in_object_store, get_pinned_object

import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

ray.init()

def load_data():
    data = pd.read_csv('./O3.csv')
    train, test = train_test_split(data, test_size=0.2)
    data_id = pin_in_object_store(data)
    train_id= pin_in_object_store(train)
    test_id = pin_in_object_store(test)
    return data_id,train_id,test_id

#load data and pin to object store
data_id, train_id, test_id = load_data()



class MLPTrainer(Trainable):
    """Example agent whose learning curve is a random sigmoid.

    The dummy hyperparameters "width" and "height" determine the slope and
    maximum reward value reached.
    """

    def _setup(self, config):
        self.timestep = 0
        self.data = get_pinned_object(data_id)
        self.train = get_pinned_object(train_id)
        self.test = get_pinned_object(test_id)
        self.hidden1 = config['hidden_layers1']
        self.lr = config['lr']
        self.max_iter = config['max_iter']

    def _train(self):
        #self.timestep += 1
        #v = np.tanh(float(self.timestep) / self.config["width"])
        #v *= self.config["height"]
        # Here we use `episode_reward_mean`, but you can also report other
        # objectives such as loss or accuracy.
        #return {"episode_reward_mean": v}

        self.timestep +=1
        model = MLPRegressor(hidden_layer_sizes=(self.hidden1,),solver='adam',learning_rate_init=self.lr,max_iter=self.max_iter)
        model.fit(self.train[['TEMPERATURE','HUMIDITY','O3']],self.train['O3_LABEL'])
        y_pred = model.predict(self.data[['TEMPERATURE','HUMIDITY','O3']])
        mse,r2=self.evaluate(self.data['O3_LABEL'],y_pred)
        return {"mse":mse, "rs":r2}

    def _save(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "w") as f:
            f.write(json.dumps({"timestep": self.timestep}))
        return path


    def _restore(self, checkpoint_path):
        with open(checkpoint_path) as f:
            self.timestep = json.loads(f.read())["timestep"]
        pass

    def evaluate(self,y_true, y_pred):
        mse = metrics.mean_squared_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
        return (mse, r2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    #ray.init()

    # Hyperband early stopping, configured with `episode_reward_mean` as the
    # objective and `training_iteration` as the time unit,
    # which is automatically filled by Tune.
    hyperband = HyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="mse",
        max_t=100)

    exp = Experiment(
        name="hyperband_test",
        run=MLPTrainer,
        num_samples=10,
        stop={"training_iteration": 1 if args.smoke_test else 99999},
        config={
            "hidden_layers1": sample_from(lambda spec: np.random.randint(32,128)),
            "lr": sample_from(lambda spec: np.random.uniform(0.001,0.1)),
            "max_iter":5000
        })

    run_experiments(exp, scheduler=hyperband)
