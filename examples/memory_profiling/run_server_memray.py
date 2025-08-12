#!/usr/bin/env python3
"""
Memory profiling wrapper for gRPC server using memray
"""
import argparse
import sys
import os
import memray
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from omegaconf import OmegaConf
from appfl.agent import ServerAgent
from appfl.comm.grpc import GRPCServerCommunicator, serve

# Optimizations are now built into the main classes via optimize_memory flags


def run_server_with_profiling(config_path: str, output_dir: str = "./memory_profiles", use_optimized: bool = False):
    """Run server with memray profiling enabled"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Choose profile file name based on version
    version_suffix = "_optimized" if use_optimized else "_original"
    profile_file = f"{output_dir}/server{version_suffix}_memory_profile.bin"
    
    with memray.Tracker(profile_file):
        # Load server configuration
        server_agent_config = OmegaConf.load(config_path)
        
        # Modify config to use optimizations if using optimized version
        if use_optimized:
            # Use VanillaTrainer with memory optimization instead of separate class
            server_agent_config.client_configs.train_configs.optimize_memory = True
            # Enable memory optimization in gRPC communicator
            server_agent_config.server_configs.comm_configs.grpc_configs.optimize_memory = True
            # Enable memory optimization in ServerAgent
            server_agent_config.server_configs.optimize_memory = True
            # Enable memory optimization in ClientAgent (passed to clients)
            server_agent_config.client_configs.optimize_memory = True
            # Enable memory optimization in Scheduler
            server_agent_config.server_configs.scheduler_kwargs.optimize_memory = True
            # Enable memory optimization in Aggregator
            server_agent_config.server_configs.aggregator_kwargs.optimize_memory = True
            print("Enabled memory optimizations for all components: Trainer, gRPC, Agents, Scheduler, and Aggregator")
        
        # Create server agent with built-in optimization flags
        server_agent = ServerAgent(server_agent_config=server_agent_config)
        optimization_status = "with memory optimizations" if use_optimized else "without optimizations"
        print(f"Server agent created {optimization_status}. Model parameters: {sum(p.numel() for p in server_agent.model.parameters()):,}")
        
        # Create gRPC communicator with built-in optimization flags
        communicator = GRPCServerCommunicator(
            server_agent,
            logger=server_agent.logger,
            **server_agent_config.server_configs.comm_configs.grpc_configs,
        )
        print(f"gRPC communicator created {optimization_status}")
        
        # Start serving
        print(f"Starting server with memory profiling. Profile will be saved to: {profile_file}")
        serve(
            communicator,
            **server_agent_config.server_configs.comm_configs.grpc_configs,
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config",
        type=str,
        default="./resources/configs/mnist/server_fedavg.yaml",
        help="Path to the server configuration file.",
    )
    argparser.add_argument(
        "--output-dir",
        type=str,
        default="./memory_profiles",
        help="Directory to save memory profiles.",
    )
    argparser.add_argument(
        "--use_optimized_version",
        action="store_true",
        help="Use optimized memory-efficient versions of agents and communicators.",
    )
    args = argparser.parse_args()
    
    run_server_with_profiling(args.config, args.output_dir, args.use_optimized_version)