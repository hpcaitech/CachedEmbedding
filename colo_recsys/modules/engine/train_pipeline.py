from typing import Optional

import torch
from torch.profiler import record_function
import wandb

from modules.gradient_handler.cowclip import cowclip
from utils.common import compute_throughput

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
        engine: torch.nn.Module,
        device: torch.device,
    ) -> None:
        self._model = engine
        self._device = device
        self._memcpy_stream: Optional[torch.cuda.streams.Stream] = (
            torch.cuda.Stream() if device.type == "cuda" else None
        )
        self._cur_batch = None
        self._label = None
        self._connected = False
        self.ave_forward_throughput = []
        self.ave_backward_throughput = []

    def _connect(self, dataloader_iter) -> None:
        cur_batch, label = next(dataloader_iter)
        self._cur_batch = cur_batch
        self._label = label
        with torch.cuda.stream(self._memcpy_stream):
            self._cur_batch = _to_device(cur_batch, self._device, non_blocking=True)
            self._label = _to_device(label, self._device, non_blocking=True)

        self._connected = True

    def progress(self, dataloader_iter):

        global args

        if not self._connected:
            self._connect(dataloader_iter)

        # Fetch next batch
        with record_function("## next_batch ##"):
            next_batch, next_label = next(dataloader_iter)
        
        cur_batch = self._cur_batch
        label = self._label
        assert cur_batch is not None and label is not None

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._model.zero_grad()

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cur_batch, self._memcpy_stream)
            _wait_for_batch(label, self._memcpy_stream)

        with record_function("## forward ##"):
            with compute_throughput(len(cur_batch)) as ctp:
                output = self._model(cur_batch)

        fwd_throughput = ctp()
        # print('forward_throughput is {:.4f}'.format(fwd_throughput))
        self.ave_forward_throughput.append(fwd_throughput)
            
        with record_function("## criterion ##"):
            losses = self._model.criterion(output, label.float())

        if args.use_wandb:
            wandb.log({'loss':losses})

        if self._model.training:
            with record_function("## backward ##"):
                with compute_throughput(len(cur_batch)) as ctp: 
                    self._model.backward(losses)
        
        bwd_throughput = ctp()
        # print('backward_throughput is {:.4f}'.format(bwd_throughput))
        self.ave_backward_throughput.append(bwd_throughput) 

        # Copy the next batch to GPU
        self._cur_batch = cur_batch = next_batch
        if self._model.training:
            self._label = label = next_label
        else:
            self._label = next_label
        with record_function("## copy_batch_to_gpu ##"):
            with torch.cuda.stream(self._memcpy_stream):
                self._cur_batch = _to_device(self._cur_batch, self._device, non_blocking=True)
                self._label = _to_device( self._label, self._device, non_blocking=True)

        # Update
        if self._model.training:
            with record_function("## optimizer ##"):
                self._model.step()

        return losses, output, label