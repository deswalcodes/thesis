import flwr as fl


def weighted_average(metrics):
    accuracies = []
    examples = []
    for client_metrics in metrics:
        accuracies.append(client_metrics[1]["accuracy"])
        examples.append(client_metrics[0])  

    total_examples = sum(examples)
    weighted_acc = sum([acc * num_examples for acc, num_examples in zip(accuracies, examples)]) / total_examples
    return {"accuracy": weighted_acc}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average  
)


fl.server.start_server(
    server_address="localhost:8081",
    config=fl.server.ServerConfig(num_rounds=11),
    strategy=strategy
)
