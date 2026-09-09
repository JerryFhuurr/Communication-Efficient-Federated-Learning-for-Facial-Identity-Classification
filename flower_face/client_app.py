"""Flower Message API handlers for local training and validation."""

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flower_face.task import Net, load_data, seed_everything, test, train as train_model
from federated_compression import (compression_settings, encode_update_with_stats,
                                   llz_settings, update_metadata)

app = ClientApp()


def setup(msg, context):
    config = context.run_config
    client_id = int(context.node_config["partition-id"])
    if int(context.node_config["num-partitions"]) != config["num-clients"]:
        raise ValueError("Launch with exactly the configured number of clients (default 4).")
    seed = config["seed"] + 1000 * int(msg.content["config"]["server-round"]) + client_id
    seed_everything(seed)
    model = Net(config["num-classes"])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    return model, config, client_id, seed


@app.train()
def train(msg: Message, context: Context):
    model, config, client_id, seed = setup(msg, context)
    method, levels = compression_settings(config)
    reference = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()} if method != "none" else None
    loader = load_data(config, client_id, "train", seed)
    loss = train_model(model, loader, config["local-epochs"], msg.content["config"]["lr"],
                       weight_decay=config.get("weight-decay", 0.0))
    content = RecordDict({"metrics": MetricRecord({"train_loss": loss, "num-examples": len(loader.dataset)})})
    if method != "none":
        server_round = int(msg.content["config"]["server-round"])
        p, window = llz_settings(config, levels=levels) if method == "qsgd-llz" else (0, 128)
        content["qsgd"], distortion = encode_update_with_stats(
            model.state_dict(), reference, levels=levels, seed=config["seed"],
            server_round=server_round, client_id=client_id, method=method,
            llz_p=p, llz_window=window)
        content["metrics"].update(distortion)
        content["update"] = ConfigRecord(update_metadata(method=method, levels=levels,
                                        server_round=server_round, llz_p=p, llz_window=window))
    else:
        content["arrays"] = ArrayRecord(model.state_dict())
    content["client"] = ConfigRecord({"partition-id": client_id})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    model, config, client_id, _ = setup(msg, context)
    loader = load_data(config, client_id, "validation")
    loss, accuracy = test(model, loader)
    content = RecordDict({"metrics": MetricRecord({
        "eval_loss": loss, "eval_acc": accuracy, "num-examples": len(loader.dataset),
    })})
    content["client"] = ConfigRecord({"partition-id": client_id})
    return Message(content=content, reply_to=msg)
