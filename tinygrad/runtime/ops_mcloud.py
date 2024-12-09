# Trying to add memory to cloud device so we do not need to transfer weights every time we start cloud server.
import binascii
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import Dict, List, Optional, Tuple

from tinygrad.device import Buffer, BufferSpec, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import DEBUG
from tinygrad.runtime.ops_cloud import (
  BufferAlloc,
  CloudDevice,
  CloudHandler,
  CloudSession,
  CopyIn,
  CopyOut,
  ProgramExec,
)
from tinygrad.tensor import Tensor


def h(d: bytes) -> str:
  binhash = hashlib.sha256(d).digest()
  return binascii.hexlify(binhash).decode()

@dataclass(frozen=True)
class ServerBuffer:
  hash: str
  id: int
  name: str


def read_server_memory(data: List[Tuple[str, List[Tuple[str, int, str]]]]) -> Dict[str, List[ServerBuffer]]:
  return {name: [ServerBuffer(hash=h, id=i, name=n) for h,i,n in bufs] for name,bufs in data}

def write_server_memory(loaded_models: Dict[str, List[ServerBuffer]]) -> List[Tuple[str, List[Tuple[str, int, str]]]]:
  return [(name, [(b.hash, b.id, b.name) for b in bufs]) for name,bufs in loaded_models.items()]

class MCloudHandler(CloudHandler):
  # TODO: support memory to be in clang or disk, and direct copy to gpu from each
  # TODO: make memory immutable and shareable across sessions
  global_buffers: Dict[str, Tuple[Buffer, int, Optional[BufferSpec], int]] = {}
  loaded_models: Dict[str, List[ServerBuffer]] = {}
  # TODO: remove two global things.
  global_tensors: List[Tensor] = []  # Needed so that associated buffers cannot be deleted.
  current_id: int = 1000000

  CloudHandler.sessions = defaultdict(lambda: CloudSession(
    programs={},
    buffers={buf[3]: (buf[0], buf[1], buf[2]) for buf in MCloudHandler.global_buffers.values()} if 'MCloudHandler' in globals() else {}
  ))

  @staticmethod
  def add_memory(models: Dict[str, Dict[str, Tensor]]):
    for model_name, weights in models.items():
      buffers = []
      for k, v in weights.items():
        assert v.device.startswith('DISK'), f"All tensors must be on disk to be added to memory, {v.device=}"
        v = v.to(MCloudHandler.device).bitcast(dtypes.uint16).cast(dtypes.uint32).mul(1 << 16).bitcast(dtypes.float32).cast(dtypes.float16).realize()
        buffer: Buffer = v.lazydata.base.buffer
        buffer._lb_refcount += 10 # TODO: remove
        data = buffer.as_buffer() 
        # TODO: take hash of disk buffer not after conversion
        hash = h(data)
        MCloudHandler.current_id += 1
        print('hash', k, hash, buffer._buf)
        MCloudHandler.global_buffers[hash] = (buffer._buf, buffer.nbytes, buffer.options, MCloudHandler.current_id)
        MCloudHandler.global_tensors.append(v)
        buffers.append(ServerBuffer(hash=hash, id=MCloudHandler.current_id, name=k))
      MCloudHandler.loaded_models[model_name] = buffers

  def setup(self):
    # move stuff in memory to device
    super().setup()
    CloudHandler.device = MCloudHandler.device
    print(f"MCloudHandler setup with {len(MCloudHandler.global_buffers)} memory")

  def _do(self, method):
    if self.path == "/renderer" and method == "GET":
      cls, args = Device[MCloudHandler.device].renderer.__reduce__()
      ret = json.dumps((cls.__module__, cls.__name__, args, write_server_memory(MCloudHandler.loaded_models))).encode()
      self.send_response(200)
      self.send_header('Content-Length', str(len(ret)))
      self.end_headers()
      return self.wfile.write(ret)
    else:
      return super()._do(method)

# Frontend
class MCloudDevice(CloudDevice):
  def __init__(self, device: str):
    print(f"INITIALIZING MCloudDevice init with device {device}")
    self.hash_to_global: Dict[str, int] = {}
    self.local_to_global: Dict[int, int] = {}
    self.loaded_models: Dict[str, List[ServerBuffer]] = {}
    super().__init__(device)

  def batch_submit(self):
    l = len(self.req._h)
    q_len = len(self.req._q)
    copy_ins = [c for c in self.req._q if isinstance(c, CopyIn) and c.datahash in self.hash_to_global]
    for c in copy_ins:
      # find corresponding alloc
      alloc_entry = [a for a in self.req._q if isinstance(a, BufferAlloc) and a.buffer_num == c.buffer_num]
      assert len(alloc_entry) == 1, f"alloc {c.buffer_num} expected 1 entry, got {len(alloc_entry)}"
      self.req._q.remove(alloc_entry[0])
      self.req._q.remove(c)
      self.req._h.pop(c.datahash)
      self.local_to_global[c.buffer_num] = self.hash_to_global[c.datahash]

    # Create new ProgramExec instances with updated bufs since ProgramExec is frozen
    for i, r in enumerate(self.req._q):
      if isinstance(r, ProgramExec):  # and any(b in self.local_to_global for b in r.bufs):
        new_bufs = tuple(self.local_to_global[b] if b in self.local_to_global else b for b in r.bufs)
        print('prev', self.req._q[i])
        self.req._q[i] = ProgramExec(r.name, r.datahash, new_bufs, r.vals, r.global_size, r.local_size, r.wait)
        print('new', self.req._q[i])
      if isinstance(r, CopyOut) and r.buffer_num in self.local_to_global:
        self.req._q[i] = CopyOut(self.local_to_global[r.buffer_num])

    if DEBUG >= 1:
      print(f'copy_ins {len(copy_ins)}, q_len {q_len} -> {len(self.req._q)}')
      print(f"MCloudDevice batch_submit: {l} -> {len(self.req._h)}")
    return super().batch_submit()

  def send(self, method, path, data: Optional[bytes] = None) -> bytes:
    response = super().send(method, path, data)
    if method == "GET" and path == "renderer":
      clouddev = json.loads(response.decode())
      print('VISH', clouddev[3])
      self.loaded_models = read_server_memory(clouddev[3])
      for model_name, saved_buffers in self.loaded_models.items():
        for buffer in saved_buffers:
          self.hash_to_global[buffer.hash] = buffer.id
      if DEBUG >= 1:
        print(f"MCloudDevice hash_to_global: {self.hash_to_global}")
    return response
