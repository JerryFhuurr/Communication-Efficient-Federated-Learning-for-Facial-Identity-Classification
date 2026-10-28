"""Flower Message API handlers for local training and validation."""

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flower_face.task import Net, load_data, seed_everything, test, train as train_model
from flower_face.task_api import resolve, fixed_state_keys, verify_fixed_state
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
    task = resolve(config, Net=Net, load_data=load_data, train=train_model, test=test)
    model = task.Net(config["num-classes"])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    return model, config, client_id, seed


@app.train()
def train(msg: Message, context: Context):
    model, config, client_id, seed = setup(msg, context)
    method, levels = compression_settings(config)
    task = resolve(config, Net=Net, load_data=load_data, train=train_model, test=test)
    fixed = fixed_state_keys(task, model)
    initial = model.state_dict()
    fixed_reference = {k: initial[k].detach().cpu().clone() for k in fixed}
    reference = {name: value.detach().cpu().clone() for name, value in initial.items()
                 if name not in fixed} if method != "none" else None
    loader = task.load_data(config, client_id, "train", seed)
    loss = task.train(model, loader, config["local-epochs"], msg.content["config"]["lr"],
                       weight_decay=config.get("weight-decay", 0.0))
    verify_fixed_state(model.state_dict(), fixed_reference)
    content = RecordDict({"metrics": MetricRecord({"train_loss": loss, "num-examples": len(loader.dataset)})})
    if method == "qsgd-llz-adaptive":
        from federated_compression.adaptive_llz import encode_reply, settings
        from flower_face.communication import measure_message
        budget, window = settings(config, levels=levels)
        return encode_reply(
            {k:v for k,v in model.state_dict().items() if k not in fixed}, reference,
            levels=levels, seed=config['seed'],
            server_round=int(msg.content['config']['server-round']), client_id=client_id,
            budget=budget, window=window, metrics=content['metrics'],
            instruction=msg, measure=measure_message)
    if method == "topk-ef":
        from federated_compression.topk import encode
        content["topk"], content["update"], stats = encode(
            {k:v for k,v in model.state_dict().items() if k not in fixed}, reference, context,
            k=config.get("topk-k",124000),
            server_round=int(msg.content["config"]["server-round"]), client_id=client_id)
        content["metrics"].update(stats)
    elif method != "none":
        server_round = int(msg.content["config"]["server-round"])
        p, window = llz_settings(config, levels=levels) if method == "qsgd-llz" else (0, 128)
        content["qsgd"], distortion = encode_update_with_stats(
            {k:v for k,v in model.state_dict().items() if k not in fixed}, reference,
            levels=levels, seed=config["seed"],
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
    task = resolve(config, Net=Net, load_data=load_data, train=train_model, test=test)
    loader = task.load_data(config, client_id, "validation")
    loss, accuracy = task.test(model, loader)
    content = RecordDict({"metrics": MetricRecord({
        "eval_loss": loss, "eval_acc": accuracy, "num-examples": len(loader.dataset),
    })})
    content["client"] = ConfigRecord({"partition-id": client_id})
    return Message(content=content, reply_to=msg)
