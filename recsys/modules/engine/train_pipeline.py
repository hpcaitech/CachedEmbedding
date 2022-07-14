from typing import Optional

import torch
import torch.nn as nn
from torch.profiler import record_function
import torchmetrics as metrics

def _to_device(batch, device: torch.device, non_blocking: bool):
    return batch.to(device=device, non_blocking=non_blocking)

def _wait_for_batch(batch, stream: Optional[torch.cuda.streams.Stream]) -> None:
    if stream is None:
        return
    torch.cuda.current_stream().wait_stream(stream)
    cur_stream = torch.cuda.current_stream()
    batch.record_stream(cur_stream)

class TrainPipelineBase:
    """
    This class runs training iterations using a pipeline of two stages, each as a CUDA
    stream, namely, the current (default) stream and `self._memcpy_stream`. For each
    iteration, `self._memcpy_stream` moves the input from host (CPU) memory to GPU
    memory, and the default stream runs forward, backward, and optimization.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        metric: Optional[metrics.Metric] = None,
    ) -> None:
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._device = device
        self._metric = metric
        self._memcpy_stream: Optional[torch.cuda.streams.Stream] = (
            torch.cuda.Stream() if device is not None and device.type == "cuda" else None
        )
        self._cur_batch = None
        self._connected = False

    def _connect(self, dataloader_iter) -> None:
        cur_batch = next(dataloader_iter)
        self._cur_batch = cur_batch
        with torch.cuda.stream(self._memcpy_stream):
            self._cur_batch = _to_device(cur_batch, self._device, non_blocking=True)
        self._connected = True

    def progress(self, dataloader_iter):
        if not self._connected:
            self._connect(dataloader_iter)

        # Fetch next batch
        with record_function("## next_batch ##"):
            next_batch = next(dataloader_iter)
        
        cur_batch = self._cur_batch
        assert cur_batch is not None

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._model.zero_grad()

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cur_batch, self._memcpy_stream)

        with record_function("## forward ##"):
            output = self._model(cur_batch.sparse_features, cur_batch.dense_features)

        with record_function("## criterion ##"):
            losses = self._criterion(output, cur_batch.labels)

        if self._model.training:
            with record_function("## backward ##"):
                losses.backward()

        # Copy the next batch to GPU
        self._cur_batch = next_batch

        with record_function("## copy_batch_to_gpu ##"):
            with torch.cuda.stream(self._memcpy_stream):
                self._cur_batch = _to_device(self._cur_batch, self._device, non_blocking=True)

        # Update
        if self._model.training:
            with record_function("## optimizer ##"):
                self._optimizer.step()

        return losses, output, cur_batch.labels