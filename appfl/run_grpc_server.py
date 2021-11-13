import hydra
from omegaconf import DictConfig

import protos.server
import protos.operator

@hydra.main(config_path="config", config_name="config")
def run_server(cfg: DictConfig) -> None:
    operator = protos.operator.FLOperator(cfg)
    operator.servicer = protos.server.FLServicer(cfg.server.id, str(cfg.server.port), operator)

    print("Starting the server to listen to requests from clients . . .")
    protos.server.serve(operator.servicer)

if __name__ == "__main__":
    run_server()