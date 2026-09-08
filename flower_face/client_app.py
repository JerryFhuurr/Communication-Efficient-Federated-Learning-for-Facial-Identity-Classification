"""Flower Message API handlers for local training and validation."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flower_face.task import Net, load_data, seed_everything, test, train as train_model

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
    loader = load_data(config, client_id, "train", seed)
    loss = train_model(model, loader, config["local-epochs"], msg.content["config"]["lr"])
    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"train_loss": loss, "num-examples": len(loader.dataset)}),
    })
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
