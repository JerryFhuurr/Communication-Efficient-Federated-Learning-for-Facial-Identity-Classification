"""Flower Message API handlers for local training and validation."""

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flower_face.task import Net, load_data, seed_everything, test, train as train_model
from flower_face.updates import CODEC, compression_settings, encode_update

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
    reference = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()} if method == "qsgd" else None
    loader = load_data(config, client_id, "train", seed)
    loss = train_model(model, loader, config["local-epochs"], msg.content["config"]["lr"])
    content = RecordDict({"metrics": MetricRecord({"train_loss": loss, "num-examples": len(loader.dataset)})})
    if method == "qsgd":
        server_round = int(msg.content["config"]["server-round"])
        content["qsgd"] = encode_update(model.state_dict(), reference, levels=levels,
                                        seed=config["seed"], server_round=server_round, client_id=client_id)
        content["update"] = ConfigRecord({"codec": CODEC, "server-round": server_round, "levels": levels})
    else:
        content["arrays"] = ArrayRecord(model.state_dict())
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    model, config, client_id, _ = setup(msg, context)
    loader = load_data(config, client_id, "validation")
    loss, accuracy = test(model, loader)
    content = RecordDict({"metrics": MetricRecord({
        "eval_loss": loss, "eval_acc": accuracy, "num-examples": len(loader.dataset),
    })})
    return Message(content=content, reply_to=msg)
