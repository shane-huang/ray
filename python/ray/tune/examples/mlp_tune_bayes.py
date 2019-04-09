"""This test checks that BayesOpt is functional.

It also checks that it is usable with a separate scheduler.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune import grid_search, sample_from

from ray.tune import run_experiments, Experiment, register_trainable
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import BayesOptSearch

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

def evaluate(y_true,y_pred):
    mse = metrics.mean_squared_error(y_true,y_pred)
    r2=metrics.r2_score(y_true,y_pred)
    return (mse,r2)


def train_nn(config, reporter):
    def get_data():
        return (
            get_pinned_object(data_id),
            get_pinned_object(train_id),
            get_pinned_object(test_id)
        )
    data,train,test=get_data()

    for i in range(config["iterations"]):
        model = MLPRegressor(hidden_layer_sizes=(int(config['hidden_layers1']),),solver='adam',learning_rate_init=config['lr'],max_iter=config['max_iter'])
        model.fit(train[['TEMPERATURE','HUMIDITY','O3']],train['O3_LABEL'])
        y_pred = model.predict(data[['TEMPERATURE','HUMIDITY','O3']])
        mse,r2=evaluate(data['O3_LABEL'],y_pred)

        reporter(
            timesteps_total=i,
            mse=mse,
            r_square=r2,
        )
    #for i in range(config["iterations"]):
    #    reporter(
    #        timesteps_total=i,
    #        neg_mean_loss=-(config["height"] - 14)**2 +
    #        abs(config["width"] - 3))
    #    time.sleep(0.02)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()

    #data = pd.read_csv("./O3.csv")


    #register_trainable("exp", train_nn)

    space = {'lr': (0.001, 0.3)}

    exp = Experiment(
        "my_exp",
        train_nn,
        num_samples=1 if args.smoke_test else 1000,
        config={
            "iterations":20,
            "max_iter": 5000,
            #"hidden_layers": (20,),
            "hidden_layers1": sample_from(
                lambda spec: np.random.randint(32,512)
            ),
            #"hidden_layers2": grid_search([4,5]),
            #"hidden_layers3": grid_search([6,7]),
            #"hidden_layers": (
            #    grid_search([1,2,3]),
            #    grid_search(range(1,50,10)),
            #    grid_search(range(1,50,10)),
            #),
        }
    )
    algo = BayesOptSearch(
        space,
        max_concurrent=4,
        reward_attr="mse",
        utility_kwargs={
            "kind": "ucb",
            "kappa": 2.5,
            "xi": 0.0
        })
    scheduler = AsyncHyperBandScheduler(reward_attr="mse")
    run_experiments(exp, search_alg=algo, scheduler=scheduler)
