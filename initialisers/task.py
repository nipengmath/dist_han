# -*- coding: utf-8 -*-
from data_loader.data_loader import TFRecordDataLoader
from utils.utils import get_args, process_config

## from models.model import RawModel
## from trainers.train import RawTrainer

#from models.example_model import Mnist
#from trainers.example_train import ExampleTrainer

from models.han_model import HanModel
from trainers.han_train import HanTrainer


def init() -> None:
    """
    The main function of the project used to initialise all the required classes
    used when training the model
    """
    # get input arguments
    args = get_args()
    # get static config information
    config = process_config()
    # combine both into dictionary
    config = {**config, **args}

    # create your data generators for each mode
    train_data = TFRecordDataLoader(config, mode="train")
    val_data = TFRecordDataLoader(config, mode="val")
    ## test_data = TFRecordDataLoader(config, mode="test")

    # initialise model
    ## model = RawModel(config)
    model = HanModel(config)

    # initialise the estimator
    ## trainer = RawTrainer(config, model, train_data, val_data, test_data)
    trainer = HanTrainer(config, model, train_data, val_data, val_data)

    # start training
    trainer.run()


if __name__ == "__main__":
    init()
