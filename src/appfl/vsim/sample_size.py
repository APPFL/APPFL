"""Report client dataset sizes to the server so aggregators can weight by them."""

from typing import Dict, List


def register_sample_sizes(server_agent, client_agents: List) -> Dict[str, int]:
    """
    Register each client's local dataset size with the server's aggregator.

    Aggregators weight an update by data size only for clients whose size has
    been registered, and silently give every other client an equal weight. Under
    ``client_weights_mode: "sample_size"`` this therefore has to run before the
    simulation starts. The drivers never call it themselves, so sizes registered
    some other way are left untouched. An agent missing either method raises
    rather than being skipped, since skipping would restore the silent fallback.

    Registration uses ``sync=False``. With ``sync=True`` each client gets a Future
    that resolves only once every client has reported, which a single process
    registering clients one at a time never reaches if it blocks on the result.

    :param server_agent: APPFL ServerAgent (or SimServerAgent) instance.
    :param client_agents: List of APPFL ClientAgent instances.
    :return: Dict mapping client_id to its registered sample size, in client order.
    """
    sizes = {client.get_id(): client.get_sample_size() for client in client_agents}
    for cid, size in sizes.items():
        server_agent.set_sample_size(client_id=cid, sample_size=size, sync=False)
    return sizes
