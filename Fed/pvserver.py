from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

import torch
import numpy as np

import torch._dynamo
torch._dynamo.config.suppress_errors = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define metric aggregation function
def custom_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}



def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


# model_params=torch.load('appleDenseNetFull.pth')
x= torch.rand(16,3,224,224)

import torchvision.models
from torch import nn


model=torchvision.models.densenet121(pretrained=False)
model.classifier=nn.Linear(model.classifier.in_features, 4)

initial_params={}
for key, value in model.state_dict().items():
    initial_params[key] = value.numpy()



x.to(device)
# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average) #MJ: replace with the strategy of your choice
# strategy = fl.server.strategy.FedMedian(evaluate_metrics_aggregation_fn=weighted_average)#,initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(model)))
# Start Flower server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=5),        #MJ: change the number of rounds here
    strategy=strategy,
)
