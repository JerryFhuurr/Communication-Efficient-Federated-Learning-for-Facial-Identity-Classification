"""FedAvg with optional QSGD delta uploads and validation-selected checkpoints."""

from copy import copy

from flwr.app import ArrayRecord, ConfigRecord, Context, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import torch

from flower_face.experiment import Experiment
from flower_face.communication import CommunicationLedger, MeasuredGrid
from flower_face.task import Net, load_data, read_manifest, seed_everything, test
from federated_compression import CODECS, compression_settings, decode_update, llz_settings
from flower_face.reproducibility import model_hash

app = ServerApp()


class CheckpointFedAvg(FedAvg):
    """Keep Flower's aggregation; record the model matching each validation round."""

    def __init__(self, experiment, clients):
        super().__init__(fraction_train=1.0, fraction_evaluate=1.0,
                         min_train_nodes=clients, min_evaluate_nodes=clients,
                         min_available_nodes=clients)
        self.experiment = experiment
        self.clients = clients
        self.current = None
        self.node_partitions = {}
        self.communication = CommunicationLedger(experiment)

    def checked_replies(self, replies):
        replies = list(replies)
        if len(replies) != self.clients or any(reply.has_error() for reply in replies):
            raise RuntimeError("A baseline round requires successful replies from all configured clients.")
        if len({reply.metadata.src_node_id for reply in replies}) != self.clients:
            raise RuntimeError("Duplicate client replies in a baseline round.")
        partitions = []
        for reply in replies:
            record = reply.content.get("client")
            partition = record.get("partition-id") if isinstance(record, ConfigRecord) else None
            if type(partition) is not int or not 0 <= partition < self.clients:
                raise RuntimeError("Every reply must identify a valid dataset partition-id")
            node = reply.metadata.src_node_id
            if node in self.node_partitions and self.node_partitions[node] != partition:
                raise RuntimeError("Client partition changed within this run")
            partitions.append(partition)
        if set(partitions) != set(range(self.clients)):
            raise RuntimeError("Replies must cover every dataset partition exactly once")
        self.node_partitions.update({r.metadata.src_node_id: p for r, p in zip(replies, partitions)})
        return [reply for _, reply in sorted(zip(partitions, replies), key=lambda item: item[0])]

    def aggregate_train(self, server_round, replies):
        arrays, metrics = super().aggregate_train(server_round, self.checked_replies(replies))
        if arrays is None or metrics is None:
            raise RuntimeError("Training aggregation returned no model or metrics.")
        self.current = (server_round, arrays, dict(metrics))
        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        metrics = super().aggregate_evaluate(server_round, self.checked_replies(replies))
        if self.current is None or self.current[0] != server_round or metrics is None:
            raise RuntimeError("Validation metrics do not match the current model round.")
        _, arrays, train_metrics = self.current
        improved = self.experiment.consider_best(
            arrays.to_torch_state_dict(), server_round, metrics["eval_loss"], metrics["eval_acc"])
        self.experiment.log_step(dict(round=server_round, train=train_metrics,
                                      validation=dict(metrics), train_clients=self.clients,
                                      validation_clients=self.clients, new_best=improved,
                                      model_sha256=model_hash(arrays.to_torch_state_dict()),
                                      communication=self.communication.summary(server_round),
                                      cumulative_communication=self.communication.summary()))
        traffic = self.communication.summary(server_round)["total"]
        print(f"Round {server_round} serialized objects: {traffic['serialized_object_bytes']:,} bytes "
              f"({traffic['serialized_object_bits']:,} bits), {traffic['messages']} messages", flush=True)
        return metrics


class QSGDFedAvg(CheckpointFedAvg):
    """Decode client deltas, average with FedAvg weights, then add to round base."""

    def __init__(self, experiment, clients, levels, *, method="qsgd", llz_p=0, llz_window=128):
        super().__init__(experiment, clients)
        self.levels = levels
        compression_settings({"compression": method, "qsgd-levels": levels,
                              "llz-p": llz_p, "llz-window": llz_window})
        if method not in CODECS:
            raise ValueError("Delta strategy requires a QSGD method")
        self.method, self.llz_p, self.llz_window = method, llz_p, llz_window
        self.reference = None
        self.reference_round = None

    def configure_train(self, server_round, arrays, config, grid):
        self.reference = {k: v.clone() for k, v in arrays.to_torch_state_dict().items()}
        self.reference_round = server_round
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        replies = self.checked_replies(replies)
        if self.reference is None or self.reference_round != server_round:
            raise RuntimeError("QSGD aggregation has no matching round reference")
        decoded_replies = []
        for reply in replies:
            delta = decode_update(reply.content, self.reference, levels=self.levels, server_round=server_round,
                                  method=self.method, llz_p=self.llz_p, llz_window=self.llz_window)
            # Grid has already logged the actual encoded message. These local
            # copies are used only to reuse Flower's sample-weighted aggregation.
            decoded = copy(reply)
            decoded.content = RecordDict({"arrays": ArrayRecord(delta), "metrics": reply.content["metrics"],
                                          "client": reply.content["client"]})
            decoded_replies.append(decoded)
        average_delta, metrics = super().aggregate_train(server_round, decoded_replies)
        delta_state = average_delta.to_torch_state_dict()
        state = {name: base + delta_state[name] for name, base in self.reference.items()}
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("QSGD aggregation produced nonfinite weights")
        arrays = ArrayRecord(state)
        self.current = (server_round, arrays, dict(metrics))
        return arrays, metrics


@app.main()
def main(grid: Grid, context: Context):
    config = dict(context.run_config)
    method, levels = compression_settings(config)
    config.update({"compression": method, "qsgd-levels": levels})
    p, window = llz_settings(config) if method == "qsgd-llz" else (0, 128)
    if method == "qsgd-llz":
        config.update({"llz-p": p, "llz-window": window})
    manifest = read_manifest(config["manifest"], config["num-classes"], config["num-clients"])
    seed_everything(config["seed"])
    model = Net(config["num-classes"])
    experiment = Experiment(config["output-dir"], "fedavg-"+method if method != "none" else "fedavg", config, manifest, "round")
    experiment.metadata["initial_model_sha256"] = model_hash(model.state_dict())
    experiment.metadata["compression"] = dict(
        method=method, codec=CODECS.get(method),
        target="client model deltas" if method != "none" else "full client models",
        normalization="one L2 norm per tensor" if method != "none" else None,
        llz_p=p if method == "qsgd-llz" else None,
        llz_window=window if method == "qsgd-llz" else None,
        downlink="uncompressed", error_feedback=False,
        rng="SeedSequence([seed, round, partition_id, 0x51534744]); sorted tensor names" if method != "none" else None)
    strategy = (QSGDFedAvg(experiment, config["num-clients"], levels, method=method, llz_p=p, llz_window=window) if method != "none"
                else CheckpointFedAvg(experiment, config["num-clients"]))
    result = strategy.start(
        grid=MeasuredGrid(grid, strategy.communication), initial_arrays=ArrayRecord(model.state_dict()),
        train_config=ConfigRecord({"lr": config["learning-rate"]}),
        num_rounds=config["num-server-rounds"], timeout=120,
    )
    expected_rounds = set(range(1, config["num-server-rounds"] + 1))
    if (set(result.train_metrics_clientapp) != expected_rounds
            or set(result.evaluate_metrics_clientapp) != expected_rounds
            or experiment.best is None):
        raise RuntimeError("Incomplete training/validation history; inspect Flower logs.")
    experiment.save_checkpoint("final_model.pt", result.arrays.to_torch_state_dict(),
                               config["num-server-rounds"],
                               dict(result.evaluate_metrics_clientapp[config["num-server-rounds"]]))
    test_metrics = None
    if config.get("evaluate-final-test", False):
        # Selection is finished. Evaluate only the validation-selected model once.
        checkpoint = torch.load(experiment.output / "best_model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        loss, accuracy = test(model, load_data(config, split="test"))
        test_metrics = dict(loss=loss, accuracy=accuracy, checkpoint="best_model.pt",
                            round=experiment.best["step"])
        print(f"Selected model held-out test: loss={loss:.4f}, accuracy={accuracy:.2%}")
    else:
        print("Held-out test skipped; use validation metrics for development.")
    experiment.complete(dict(
        config=config, image_variant=manifest["image_variant"],
        versions=experiment.metadata["versions"], manifest_sha256=experiment.manifest_hash,
        identities=manifest["identities"],
        train={str(k): dict(v) for k, v in result.train_metrics_clientapp.items()},
        validation={str(k): dict(v) for k, v in result.evaluate_metrics_clientapp.items()},
        test=test_metrics, model_parameters=sum(p.numel() for p in model.parameters()),
        final_checkpoint={"checkpoint": "final_model.pt", "round": config["num-server-rounds"]},
        communication=strategy.communication.summary(),
    ))
    traffic = strategy.communication.summary()
    print(f"Logical serialized communication: {traffic['total']['serialized_object_bytes']:,} bytes "
          f"({traffic['total']['serialized_object_bits']:,} bits); "
          f"downlink={traffic['downlink']['serialized_object_bytes']:,}, "
          f"uplink={traffic['uplink']['serialized_object_bytes']:,}. Not network traffic.")
    print(f"Saved best and final checkpoints, metrics, and manifest to {experiment.output.resolve()}")
