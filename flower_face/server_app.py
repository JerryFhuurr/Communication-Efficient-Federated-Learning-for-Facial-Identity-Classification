"""FedAvg with optional QSGD delta uploads and validation-selected checkpoints."""

from copy import copy
import math
import json
from time import perf_counter

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import torch

from flower_face.experiment import Experiment
from flower_face.communication import CommunicationLedger, MeasuredGrid, measure_message
from flower_face.task import Net, load_data, read_manifest, seed_everything, test
from flower_face.task_api import resolve, fixed_state_keys, verify_fixed_state
from federated_compression import CODECS, compression_settings, decode_update, llz_settings
from flower_face.reproducibility import model_hash

app = ServerApp()

LLZ_CLIENT_METRICS = (
    "llz_code_squared_error",
    "llz_reconstruction_squared_error",
    "llz_qsgd_squared_norm",
    "llz_max_code_error",
    "llz_symbol_count",
)
QSGD_CLIENT_METRICS = (
    "qsgd_quantization_squared_error",
    "qsgd_input_squared_norm",
    "qsgd_max_abs_error",
    "qsgd_coordinate_count",
)


def aggregate_qsgd_distortion(replies):
    """Aggregate QSGD error against the original full-precision client deltas."""
    totals = {key: 0.0 for key in QSGD_CLIENT_METRICS}
    maximum = 0.0
    for reply in replies:
        metrics = reply.content["metrics"]
        if any(key not in metrics for key in QSGD_CLIENT_METRICS):
            raise ValueError("Compressed reply is missing QSGD distortion measurements")
        for key in QSGD_CLIENT_METRICS:
            value = metrics[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid QSGD distortion measurement: {key}")
            totals[key] += value
        if type(metrics["qsgd_coordinate_count"]) is not int:
            raise ValueError("QSGD coordinate count must be an integer")
        maximum = max(maximum, metrics["qsgd_max_abs_error"])
    coordinates = int(totals["qsgd_coordinate_count"])
    if coordinates <= 0:
        raise ValueError("QSGD coordinate count must be positive")
    squared = totals["qsgd_quantization_squared_error"]
    reference = totals["qsgd_input_squared_norm"]
    return dict(
        qsgd_clients=len(replies),
        qsgd_coordinate_count=coordinates,
        qsgd_max_abs_error=maximum,
        qsgd_quantization_mse=squared / coordinates,
        qsgd_quantization_squared_error=squared,
        qsgd_input_squared_norm=reference,
        qsgd_relative_squared_error=(squared / reference if reference else 0.0),
    )


def aggregate_llz_distortion(replies, p):
    """Aggregate codec distortion across transmitted coordinates, not examples."""
    totals = {key: 0.0 for key in LLZ_CLIENT_METRICS}
    maximum = 0
    for reply in replies:
        metrics = reply.content["metrics"]
        if any(key not in metrics for key in LLZ_CLIENT_METRICS):
            raise ValueError("QSGD+LLZ reply is missing distortion measurements")
        for key in LLZ_CLIENT_METRICS:
            value = metrics[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid QSGD+LLZ distortion measurement: {key}")
            totals[key] += value
        observed_max = metrics["llz_max_code_error"]
        symbols = metrics["llz_symbol_count"]
        if type(observed_max) is not int or type(symbols) is not int or symbols <= 0:
            raise ValueError("LLZ maximum code error and symbol count must be integers")
        maximum = max(maximum, observed_max)
    if maximum > p:
        raise ValueError("Reported LLZ code error exceeds configured p")
    if p == 0 and (totals["llz_code_squared_error"] != 0
                   or totals["llz_reconstruction_squared_error"] != 0):
        raise ValueError("Lossless LLZ reported nonzero secondary distortion")
    symbols = int(totals["llz_symbol_count"])
    squared = totals["llz_reconstruction_squared_error"]
    reference = totals["llz_qsgd_squared_norm"]
    return dict(
        llz_p=p,
        llz_clients=len(replies),
        llz_symbol_count=symbols,
        llz_max_code_error=maximum,
        llz_code_mse=totals["llz_code_squared_error"] / symbols,
        llz_reconstruction_squared_error=squared,
        llz_qsgd_squared_norm=reference,
        llz_relative_squared_error=(squared / reference if reference else 0.0),
    )


class CheckpointFedAvg(FedAvg):
    """Keep Flower's aggregation; record the model matching each validation round."""

    def __init__(self, experiment, clients, *, fixed_state=None):
        super().__init__(fraction_train=1.0, fraction_evaluate=1.0,
                         min_train_nodes=clients, min_evaluate_nodes=clients,
                         min_available_nodes=clients)
        self.experiment = experiment
        self.clients = clients
        self.current = None
        self.node_partitions = {}
        self.communication = CommunicationLedger(experiment)
        self.fixed_state = {k:v.detach().cpu().clone() for k,v in (fixed_state or {}).items()}

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
        replies = self.checked_replies(replies)
        if self.fixed_state:
            for reply in replies:
                verify_fixed_state(reply.content['arrays'].to_torch_state_dict(), self.fixed_state)
        arrays, metrics = super().aggregate_train(server_round, replies)
        if arrays is None or metrics is None:
            raise RuntimeError("Training aggregation returned no model or metrics.")
        if self.fixed_state:
            state = arrays.to_torch_state_dict()
            state.update({k:v.clone() for k,v in self.fixed_state.items()})
            arrays = ArrayRecord(state)
        self.current = (server_round, arrays, dict(metrics))
        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        metrics = super().aggregate_evaluate(server_round, self.checked_replies(replies))
        if self.current is None or self.current[0] != server_round or metrics is None:
            raise RuntimeError("Validation metrics do not match the current model round.")
        _, arrays, train_metrics = self.current
        improved = self.experiment.consider_best(
            arrays.to_torch_state_dict(), server_round, metrics["eval_loss"], metrics["eval_acc"])
        timing = {}
        if getattr(self, 'method', None) == 'topk-ef':
            now=perf_counter()
            timing=dict(round_seconds=now-self.round_started,
                        training_phase_seconds=self.train_finished-self.round_started,
                        validation_phase_seconds=now-self.train_finished)
        self.experiment.log_step(dict(timing=timing, round=server_round, train=train_metrics,
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

    def __init__(self, experiment, clients, levels, *, method="qsgd", llz_p=0, llz_window=128, fixed_state=None, topk_k=124000, llz_budget=0.0):
        # Delta aggregation contains no fixed buffers. Restore them after averaging.
        super().__init__(experiment, clients)
        self.fixed_reference = {k:v.detach().cpu().clone() for k,v in (fixed_state or {}).items()}
        self.levels = levels
        self.topk_k = topk_k
        self.llz_budget = llz_budget
        compression_settings({"compression": method, "qsgd-levels": levels,
                              "llz-p": llz_p, "llz-window": llz_window, "llz-budget": llz_budget})
        if method not in CODECS:
            raise ValueError("Delta strategy requires a QSGD method")
        self.method, self.llz_p, self.llz_window = method, llz_p, llz_window
        self.reference = None
        self.reference_round = None

    def configure_train(self, server_round, arrays, config, grid):
        if self.method == "topk-ef":
            self.round_started = perf_counter()
        state = arrays.to_torch_state_dict()
        verify_fixed_state(state, self.fixed_reference)
        self.reference = {k: v.clone() for k, v in state.items() if k not in self.fixed_reference}
        self.reference_round = server_round
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        replies = self.checked_replies(replies)
        if self.reference is None or self.reference_round != server_round:
            raise RuntimeError("QSGD aggregation has no matching round reference")
        distortion = {} if self.method == "topk-ef" else aggregate_qsgd_distortion(replies)
        timing_rows = []
        adaptive_rows = []
        if self.method == "qsgd-llz":
            distortion.update(aggregate_llz_distortion(replies, self.llz_p))
        decoded_replies = []
        for reply in replies:
            if self.method == "topk-ef":
                from federated_compression.topk import decode
                started = perf_counter()
                delta = decode(reply.content, self.reference, k=self.topk_k, server_round=server_round)
                decode_seconds = perf_counter()-started
                observed = {key: value for key,value in reply.content['metrics'].items() if key.startswith('topk_')}
                required = {'topk_encode_seconds','topk_diagnostic_seconds','topk_corrected_squared_norm',
                            'topk_residual_squared_norm','topk_delta_squared_norm','topk_transmitted_vs_delta_squared_error'}
                if set(observed)!=required or any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in observed.values()):
                    raise ValueError('Invalid Top-k diagnostic/timing metrics')
                timing_rows.append(dict(round=server_round, client=reply.content['client']['partition-id'],
                                        decode_seconds=decode_seconds, **observed))
            elif self.method == 'qsgd-llz-adaptive':
                from federated_compression.adaptive_llz import decode
                started = perf_counter()
                delta = decode(reply.content, self.reference, levels=self.levels,
                    server_round=server_round, budget=self.llz_budget, window=self.llz_window)
                adaptive_rows.append(dict(round=server_round,
                    client=reply.content['client']['partition-id'],
                    decode_seconds=perf_counter()-started,
                    observed_message_bytes=measure_message(reply)['serialized_object_bytes'],
                    **dict(reply.content['metrics'])))
            else:
                delta = decode_update(reply.content, self.reference, levels=self.levels, server_round=server_round,
                                      method=self.method, llz_p=self.llz_p, llz_window=self.llz_window)
            # Grid has already logged the actual encoded message. These local
            # copies are used only to reuse Flower's sample-weighted aggregation.
            decoded = copy(reply)
            aggregation_metrics = MetricRecord({
                "train_loss": reply.content["metrics"]["train_loss"],
                "num-examples": reply.content["metrics"]["num-examples"],
            })
            decoded.content = RecordDict({"arrays": ArrayRecord(delta), "metrics": aggregation_metrics,
                                          "client": reply.content["client"]})
            decoded_replies.append(decoded)
        average_delta, metrics = super().aggregate_train(server_round, decoded_replies)
        delta_state = average_delta.to_torch_state_dict()
        state = {name: base + delta_state[name] for name, base in self.reference.items()}
        state.update({k:v.clone() for k,v in self.fixed_reference.items()})
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("QSGD aggregation produced nonfinite weights")
        arrays = ArrayRecord(state)
        metrics.update(distortion)
        if self.method == 'qsgd-llz-adaptive':
            distortion = aggregate_llz_distortion(replies, p=2)
            distortion.pop('llz_p')  # Each client can choose a different p.
            metrics.update(distortion)
            for p in (0, 1, 2):
                metrics[f'adaptive_p{p}_clients'] = sum(row['adaptive_selected_p']==p for row in adaptive_rows)
            norm = sum(row['qsgd_input_squared_norm'] for row in adaptive_rows)
            extra = sum(row['llz_reconstruction_squared_error'] for row in adaptive_rows)
            metrics['adaptive_extra_relative_squared_error'] = extra/norm if norm else 0.0
            metrics['adaptive_selection_seconds_sum'] = sum(row['adaptive_selection_seconds'] for row in adaptive_rows)
            with (self.experiment.output/'adaptive_selection.jsonl').open('a',encoding='utf-8') as stream:
                for row in adaptive_rows:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
        if self.method == 'topk-ef':
            with (self.experiment.output/'codec_timing.jsonl').open('a',encoding='utf-8') as stream:
                for row in timing_rows:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
            self.train_finished = perf_counter()
        self.current = (server_round, arrays, dict(metrics))
        return arrays, metrics


@app.main()
def main(grid: Grid, context: Context):
    config = dict(context.run_config)
    method, levels = compression_settings(config)
    config.update({"compression": method, "qsgd-levels": levels})
    p, window = llz_settings(config, levels=levels) if method == "qsgd-llz" else (0, 128)
    budget = 0.0
    if method == 'qsgd-llz-adaptive':
        from federated_compression.adaptive_llz import settings
        budget, window = settings(config, levels=levels)
        config.update({'llz-budget': budget, 'llz-window': window})
    if method == "qsgd-llz":
        config.update({"llz-p": p, "llz-window": window})
    task = resolve(config, Net=Net, load_data=load_data, read_manifest=read_manifest, test=test)
    manifest = task.read_manifest(config["manifest"], config["num-classes"], config["num-clients"])
    seed_everything(config["seed"])
    model = task.Net(config["num-classes"])
    fixed_keys = fixed_state_keys(task, model)
    fixed_state = {k:v for k,v in model.state_dict().items() if k in fixed_keys}
    experiment = Experiment(config["output-dir"], "fedavg-"+method if method != "none" else "fedavg", config, manifest, "round")
    experiment.metadata["initial_model_sha256"] = model_hash(model.state_dict())
    experiment.metadata['fixed_state_policy'] = dict(
        policy='immutable server-initial buffers', names=sorted(fixed_keys),
        upload='full buffers with uncompressed models; omitted from compressed deltas',
        downlink='included in every full-state download')
    experiment.metadata["compression"] = dict(
        method=method, codec=CODECS.get(method),
        target="client model deltas" if method != "none" else "full client models",
        normalization="one L2 norm per tensor" if method in ("qsgd","qsgd-llz","qsgd-llz-adaptive") else None,
        llz_p=p if method == "qsgd-llz" else None,
        llz_window=window if method in ("qsgd-llz", "qsgd-llz-adaptive") else None,
        qsgd_distortion=("summed over all client tensor coordinates relative to the "
                         "original full-precision non-fixed model deltas; reported in aggregated train metrics"
                         if method in ("qsgd","qsgd-llz","qsgd-llz-adaptive") else None),
        llz_distortion=("summed over all client tensor coordinates relative to the "
                        "dequantized QSGD updates; reported in aggregated train metrics"
                        if method in ("qsgd-llz", "qsgd-llz-adaptive") else None),
        downlink="uncompressed", error_feedback=method == "topk-ef",
        rng="SeedSequence([seed, round, partition_id, 0x51534744]); sorted tensor names" if method in ("qsgd","qsgd-llz","qsgd-llz-adaptive") else None)
    if method == 'qsgd-llz-adaptive':
        experiment.metadata['compression']['adaptive'] = dict(
            candidates=[0,1,2], budget=budget, scope='one p per whole client update',
            constraint='sum(||decode(LLZ(QSGD(u)))-decode(QSGD(u))||^2) / sum(||u||^2) <= budget; zero norm requires zero error',
            objective='minimum serialized reply object graph bytes, including diagnostics; lower p breaks ties',
            fallback='p=0; fail on codec error rather than relax budget',
            diagnostics='adaptive_selection.jsonl; preflight client bytes and observed server bytes separately',
            error_feedback=False)
    if method == "topk-ef":
        experiment.metadata["compression"].update(topk_k=config.get("topk-k",124000),
            selection="global magnitude; ascending index breaks ties",
            residual="float32 per client in Context.state; sequential rounds required; no model-only resume")
        experiment.metadata["timing"] = dict(
            client_encode="layout validation, delta, residual, selection, pack, local decode and state persistence",
            diagnostic="squared-norm calculations separately timed",
            server_decode="validation, unpack and tensor conversion, excluding aggregation",
            round="configure_train through validation/checkpoint selection; excludes history writing",
            warmup="none excluded; report first round separately",
            limitation="CPU simulation wall times; no network latency or historical codec speed comparison")
    strategy = (QSGDFedAvg(experiment, config["num-clients"], levels, method=method, llz_p=p,
                          llz_window=window, fixed_state=fixed_state, topk_k=config.get("topk-k",124000),
                          llz_budget=budget) if method != "none"
                else CheckpointFedAvg(experiment, config["num-clients"], fixed_state=fixed_state))
    result = strategy.start(
        grid=MeasuredGrid(grid, strategy.communication), initial_arrays=ArrayRecord(model.state_dict()),
        train_config=ConfigRecord({"lr": config["learning-rate"]}),
        num_rounds=config["num-server-rounds"], timeout=getattr(task, 'ROUND_TIMEOUT', 120),
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
        loss, accuracy = task.test(model, task.load_data(config, split="test"))
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
