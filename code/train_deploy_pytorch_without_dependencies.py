import argparse
import numpy as np
import os
import sys
import logging
import json
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_model_def import get_model
from d2l import torch as d2l


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    """
    Parse arguments passed from the SageMaker API
    to the container
    """

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.1)

    # Data directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # Model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_known_args()


def get_train_data(train_dir):
    """
    Get the training data and convert to tensors
    """

    x_train = np.load(os.path.join(train_dir, "x_train.npy"))
    y_train = np.load(os.path.join(train_dir, "y_train.npy"))
    logger.info(f"x train: {x_train.shape}, y train: {y_train.shape}")

    return torch.from_numpy(x_train), torch.from_numpy(y_train)


def get_test_data(test_dir):
    """
    Get the testing data and convert to tensors
    """

    x_val = np.load(os.path.join(test_dir, "x_val.npy"))
    y_val = np.load(os.path.join(test_dir, "y_val.npy"))
    logger.info(f"x val: {x_val.shape}, y val: {y_val.shape}")

    return torch.from_numpy(x_val), torch.from_numpy(y_val)


def model_fn(model_dir):
    """
    Load the model for inference
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.load_state_dict(torch.load(model_dir + "/model.pth"))
    model.eval()
    return model.to(device)


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """

    if request_content_type == "application/json":
        request = json.loads(request_body)
        train_inputs = torch.tensor(request)
        return train_inputs


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        return model(input_data).numpy()[0]


def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """Compute the accuracy for a model on a dataset using a GPU."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            # Required for BERT Fine-tuning (to be covered later)
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]


def train():
    """
    Train the PyTorch model
    """
    x_train, y_train = get_train_data(args.train)
    x_test, y_test = get_test_data(args.test)
    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    logger.info(
        "batch_size = {}, epochs = {}, learning rate = {}".format(batch_size, epochs, learning_rate)
    )

    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size, shuffle=True)

    model = get_model()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for x_train_batch, y_train_batch in train_dl:
            x, y = x_train_batch.to(device), y_train_batch.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch += 1
        logger.info(f"epoch: {epoch} -> loss: {loss}")

    # evalutate on validation set
    test_acc = evaluate_accuracy_gpu(model, test_dl)
    logger.info(f"Validation loss: -> {test_acc}")

    torch.save(model.state_dict(), args.model_dir + "/model.pth")
    # PyTorch requires that the inference script must
    # be in the .tar.gz model file and Step Functions SDK doesn't do this.
    inference_code_path = args.model_dir + "/code/"

    if not os.path.exists(inference_code_path):
        os.mkdir(inference_code_path)
        logger.info("Created a folder at {}!".format(inference_code_path))

    shutil.copy("train_deploy_pytorch_without_dependencies.py", inference_code_path)
    shutil.copy("pytorch_model_def.py", inference_code_path)
    logger.info("Saving models files to {}".format(inference_code_path))


if __name__ == "__main__":

    args, _ = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("GPU test: {}".format(device))
    train()