"""Message tags and kinds of the vsim MPI runtime.

One duplicated communicator carries everything. Control messages are small pickled
dicts on :data:`TAG_CTRL`. Model tensors travel as raw byte buffers produced by
:mod:`appfl.vsim.codec`: global models on :data:`TAG_MODEL` (coordinator to worker)
and local models on :data:`TAG_RESULT` (worker to coordinator). Each payload's
schema rides in the control message that announces it.

Setup uses blocking collectives in a fixed order: ``bcast`` of the HELLO dict,
``Bcast`` of the initial model, ``gather`` of each worker's READY or ERROR reply.

Coordinator to worker:

- ``TASK`` ``{tid, cid, version, deadline, seed[, schema, nbytes]}``: queue local
  training. A ``TAG_MODEL`` payload follows only when ``schema`` is present.
- ``BCAST`` ``{version, schema, nbytes}``: the worker joins an ``Ibcast`` of that
  global model (sync rounds publish each version once to every worker).
- ``FETCH`` ``{tid}``: send the held result on ``TAG_RESULT``, then free it.
- ``DROP`` ``{tid}``: forget a queued task or held result. A ``DONE`` already on its
  way is ignored by the coordinator.
- ``SHUTDOWN``: drop queued work, finish in-flight sends, reply ``BYE``.

Worker to coordinator:

- ``DONE`` ``{tid, meta, schema, nbytes, busy}``: a result is held and can be fetched.
- ``ERROR`` ``{tid, rank, traceback}``: training or a protocol step failed.
- ``BYE`` ``{stats}``: the worker's last message.

Matching relies on MPI's non-overtaking rule. A worker posts its ``TAG_MODEL``
receives in the order it reads TASK headers, and the coordinator posts its
``TAG_RESULT`` receives for one worker in the order it sends that worker FETCHes.
"""

TAG_CTRL = 11
TAG_MODEL = 12
TAG_RESULT = 13

HELLO = "hello"
READY = "ready"
TASK = "task"
BCAST = "bcast"
FETCH = "fetch"
DROP = "drop"
SHUTDOWN = "shutdown"
DONE = "done"
ERROR = "error"
BYE = "bye"

# Raw payloads go out as one MPI message with an int32 element count.
MAX_PAYLOAD_BYTES = 2**31 - 1


def check_payload_size(nbytes: int) -> None:
    """Raise if a payload is too large for one MPI message."""
    if nbytes > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"Model payload of {nbytes / 2**30:.2f} GiB exceeds the 2 GiB limit of a "
            f"single MPI message. Larger models need chunked transfers, which the "
            f"vsim MPI runtime does not implement yet."
        )
