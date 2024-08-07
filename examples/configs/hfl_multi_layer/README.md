# Multi-Layer HFL

## Topology

<p align="center">
  <img src='hfl-multilayer.jpg' style="width: 85%; height: auto;"/>
</p>

## How to Run?

Open first terminal to launch the server
```bash
python hfl/run_server.py --config ./configs/hfl_multi_layer/server.yaml
```

Open the second terminal to launch node 0
```bash
python hfl/run_node.py --config ./configs/hfl_multi_layer/node_0.yaml
```

Open the third terminal to launch node 1
```bash
python hfl/run_node.py --config ./configs/hfl_multi_layer/node_1.yaml
```

Open fourth terminal to launch node 2
```bash
python hfl/run_node.py --config ./configs/hfl_multi_layer/node_2.yaml
```

Open fifth terminal to launch node 3
```bash
python hfl/run_node.py --config ./configs/hfl_multi_layer/node_3.yaml
```

Finally, you can open the sixth terminal to run all seven clients together
```bash
python hfl/run_client.py --config ./configs/hfl_multi_layer/client_0.yaml &
python hfl/run_client.py --config ./configs/hfl_multi_layer/client_1.yaml &
python hfl/run_client.py --config ./configs/hfl_multi_layer/client_2.yaml &
python hfl/run_client.py --config ./configs/hfl_multi_layer/client_3.yaml &
python hfl/run_client.py --config ./configs/hfl_multi_layer/client_4.yaml &
python hfl/run_client.py --config ./configs/hfl_multi_layer/client_5.yaml &
python hfl/run_client.py --config ./configs/hfl_multi_layer/client_6.yaml &
```

## Run Together

Of course, you can run all in open terminal (though this means that the outputs for all will be shown in a single terminal and can be a bit messy...)

```bash
python hfl/run_server.py --config ./configs/hfl_multi_layer/server.yaml &
python hfl/run_node.py --config ./configs/hfl_multi_layer/node_0.yaml &
python hfl/run_node.py --config ./configs/hfl_multi_layer/node_1.yaml &
python hfl/run_node.py --config ./configs/hfl_multi_layer/node_2.yaml &
python hfl/run_node.py --config ./configs/hfl_multi_layer/node_3.yaml &
python hfl/run_client.py --config ./configs/hfl_multi_layer/client_0.yaml &
python hfl/run_client.py --config ./configs/hfl_multi_layer/client_1.yaml &
python hfl/run_client.py --config ./configs/hfl_multi_layer/client_2.yaml &
python hfl/run_client.py --config ./configs/hfl_multi_layer/client_3.yaml &
python hfl/run_client.py --config ./configs/hfl_multi_layer/client_4.yaml &
python hfl/run_client.py --config ./configs/hfl_multi_layer/client_5.yaml &
python hfl/run_client.py --config ./configs/hfl_multi_layer/client_6.yaml &
```