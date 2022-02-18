import os
from typing import Tuple
import pandas
from simpletransformers.ner import NERModel, NERArgs


def train(model_type: Tuple[str, str], num_train_epochs: int, seed: int):
    """
    Loads the pretrained model, trains it on the train set for the given seed and saves it to /trained_models

    :param model_type: tuple (general model type, official transformer model_id on huggingface)
    :param num_train_epochs: number of epochs to train the model for
    :param seed: seed to use for model training
    """

    # load train, val and test set and convert to str to avoid problems with numeric tokens that
    # may be auto-converted to int
    train_data = pandas.read_excel("training_data/df_train.xlsx", engine="openpyxl").dropna()
    train_data['words'] = train_data['words'].astype(str)
    val_data = pandas.read_excel("training_data/df_val.xlsx", engine="openpyxl").dropna()
    val_data['words'] = val_data['words'].astype(str)
    test_data = pandas.read_excel("training_data/df_test.xlsx", engine="openpyxl").dropna()
    test_data['words'] = test_data['words'].astype(str)

    path = "trained_models/{}".format(model_type[1].replace("/", "_"))

    # create sequence tagger args object to manually set model arguments
    model_args = NERArgs()
    # set seed
    model_args.manual_seed = seed
    # set batch size for training and evaluation
    model_args.train_batch_size = 8
    model_args.eval_batch_size = 8
    # skip detailed evaluation during training to make training quicker
    model_args.evaluate_during_training = False
    model_args.evaluate_during_training_verbose = False
    # deactivate multiprocessing as it causes problems in combination with gpu usage
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    # do not save detailed evaluation and full model after each epoch or checkpoint
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = False
    # save the best model to /best_model once training is done
    model_args.save_best_model = True
    # print classification report once the training is done
    model_args.classification_report = True
    # set custom output path
    model_args.output_dir = path
    # limit training epochs to num_train_epochs
    model_args.num_train_epochs = num_train_epochs
    # hand labels to train to model
    model_args.labels_list = ["O", "B-positive", "I-positive", "B-negative", "I-negative",
                              "B-neutral", "I-neutral", "B-conflict", "I-conflict"]

    if os.path.exists(path):
        print("SKIPPED: ", model_type[1].replace("/", "_"), seed)
        return

    # hand base model, model path and arguments to model
    model = NERModel(
        model_type[0], model_type[1], args=model_args
    )

    # train the model
    model.train_model(train_data, eval_data=val_data)

    # print evaluation once training is done
    result, model_outputs, preds_list = model.eval_model(test_data)
    print(result)


if __name__ == '__main__':
    train(model_type=("deberta-v2", "microsoft/deberta-v3-base"), num_train_epochs=10, seed=1)
