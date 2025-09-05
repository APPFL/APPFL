#!/usr/bin/env python3
"""
APPFL TES Federated Learning Integration Test

This script tests a complete federated learning workflow using TES.
It uses synthetic data to avoid external dependencies.
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, Any

# Add APPFL to path
sys.path.insert(0, '../../src')

def create_synthetic_mnist_data(num_samples=100):
    """Create synthetic MNIST-like data for testing."""
    # Generate random 28x28 images
    X = torch.randn(num_samples, 1, 28, 28)
    # Generate random labels (0-9)
    y = torch.randint(0, 10, (num_samples,))
    return X, y

class MockMNISTDataset:
    """Mock MNIST dataset for testing."""
    def __init__(self, num_samples=100):
        self.data, self.targets = create_synthetic_mnist_data(num_samples)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class SimpleCNN(torch.nn.Module):
    """Simple CNN model for testing."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)
        
    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

def test_mock_federated_learning():
    """Test federated learning workflow with mock components."""
    print("🔄 Testing mock federated learning workflow...")
    
    try:
        from appfl.config import ServerAgentConfig, ClientAgentConfig
        from appfl.agent import ServerAgent
        from appfl.comm.tes import TESServerCommunicator
        import pickle
        import tempfile
        
        # Create test configuration
        config_dict = {
            'server_configs': {
                'scheduler': 'SyncScheduler',
                'aggregator': 'FedAvgAggregator',
                'num_global_epochs': 2,
                'comm_configs': {
                    'tes_configs': {
                        'tes_endpoint': os.getenv('TES_ENDPOINT', 'http://localhost:8000'),
                        'docker_image': 'python:3.8-slim',  # Use simple image for testing
                        'resource_requirements': {
                            'cpu_cores': 1,
                            'ram_gb': 1.0,
                            'disk_gb': 2.0
                        }
                    }
                },
                'aggregator_kwargs': {'num_clients': 2},
                'scheduler_kwargs': {'num_clients': 2}
            },
            'client_configs': {
                'train_configs': {
                    'trainer': 'VanillaTrainer',
                    'num_local_steps': 2
                },
                'model_configs': {'model': 'CNN'},
                'data_configs': {'dataset': 'MNIST', 'batch_size': 32}
            }
        }
        
        # Convert to OmegaConf structure
        config_omega = OmegaConf.create(config_dict)
        server_config = ServerAgentConfig(**config_omega)
        
        # Create client configurations
        client_configs = []
        for i in range(2):
            client_config = ClientAgentConfig(**config_omega['client_configs'])
            client_config.client_id = f"test_client_{i+1}"
            client_configs.append(client_config)
        
        # Test TES communicator creation
        print("Creating TES communicator...")
        tes_comm = TESServerCommunicator(
            server_agent_config=server_config,
            client_agent_configs=client_configs
        )
        print("✅ TES communicator created successfully")
        
        # Test task creation (without actual submission)
        print("Testing task creation...")
        model_data = pickle.dumps({'test': 'model'})
        metadata = {'epoch': 1, 'test': True}
        
        tes_task = tes_comm._create_tes_task(
            client_id="test_client_1",
            task_name="train",
            model_data=model_data,
            metadata=metadata
        )
        
        print("✅ TES task created successfully")
        print(f"   Task name: {tes_task['name']}")
        print(f"   Executors: {len(tes_task['executors'])}")
        print(f"   Inputs: {len(tes_task['inputs'])}")
        print(f"   Outputs: {len(tes_task['outputs'])}")
        
        # Cleanup
        tes_comm.shutdown_all_clients()
        
        return True
        
    except Exception as e:
        print(f"❌ Mock federated learning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tes_task_with_simple_command():
    """Test TES task execution with a simple Python command."""
    print("🔄 Testing TES task with simple Python command...")
    
    try:
        from appfl.config import ServerAgentConfig, ClientAgentConfig
        from appfl.comm.tes import TESServerCommunicator
        import requests
        
        # Create minimal config
        config_dict = {
            'server_configs': {
                'scheduler': 'SyncScheduler',
                'aggregator': 'FedAvgAggregator',
                'comm_configs': {
                    'tes_configs': {
                        'tes_endpoint': os.getenv('TES_ENDPOINT', 'http://localhost:8000'),
                        'docker_image': 'python:3.8-slim',
                        'resource_requirements': {
                            'cpu_cores': 1,
                            'ram_gb': 1.0,
                            'disk_gb': 1.0
                        }
                    }
                },
                'aggregator_kwargs': {'num_clients': 1},
                'scheduler_kwargs': {'num_clients': 1}
            },
            'client_configs': {
                'train_configs': {'trainer': 'VanillaTrainer'},
                'model_configs': {'model': 'CNN'},
                'data_configs': {'dataset': 'MNIST'}
            }
        }
        
        server_config = ServerAgentConfig(**config_dict)
        client_config = ClientAgentConfig(**config_dict['client_configs'])
        client_config.client_id = "test_client_1"
        
        tes_comm = TESServerCommunicator(
            server_agent_config=server_config,
            client_agent_configs=[client_config]
        )
        
        # Create a simple test task that mimics APPFL client behavior
        test_task = {
            "name": "appfl-mock-training-task",
            "description": "Mock APPFL training task for testing",
            "inputs": [],
            "outputs": [{
                "name": "training_result",
                "path": "/tmp/training_result.json",
                "type": "FILE"
            }],
            "executors": [{
                "image": "python:3.8-slim",
                "command": [
                    "python", "-c", 
                    "import json; "
                    "result = {'client_id': 'test_client_1', 'loss': 0.5, 'accuracy': 0.8}; "
                    "open('/tmp/training_result.json', 'w').write(json.dumps(result))"
                ],
                "workdir": "/tmp"
            }],
            "resources": {
                "cpu_cores": 1,
                "ram_gb": 1.0,
                "disk_gb": 1.0
            }
        }
        
        # Submit task directly to TES
        tes_task_id = tes_comm._submit_tes_task(test_task)
        print(f"✅ Test task submitted with ID: {tes_task_id}")
        
        # Monitor task (with timeout)
        print("Monitoring task execution...")
        max_wait = 60  # 1 minute timeout
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            task_info = tes_comm._get_tes_task_status(tes_task_id)
            state = task_info.get("state", "UNKNOWN")
            print(f"Task state: {state}")
            
            if state == "COMPLETE":
                print("✅ Test task completed successfully!")
                tes_comm.shutdown_all_clients()
                return True
            elif state in ["SYSTEM_ERROR", "EXECUTOR_ERROR", "CANCELED"]:
                print(f"❌ Test task failed with state: {state}")
                if "logs" in task_info:
                    print(f"Logs: {task_info['logs']}")
                tes_comm.shutdown_all_clients()
                return False
            
            time.sleep(3)
        
        print("❌ Test task timed out")
        tes_comm.shutdown_all_clients()
        return False
        
    except Exception as e:
        print(f"❌ TES task test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run federated learning integration tests."""
    print("🧪 APPFL TES Federated Learning Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Mock Federated Learning Workflow", test_mock_federated_learning),
        ("TES Task with Simple Command", test_tes_task_with_simple_command)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All federated learning tests passed!")
        print("\n📝 Next steps:")
        print("   1. Build APPFL client Docker image")
        print("   2. Test with real federated learning workload")
        print("   3. Deploy to production TES endpoint")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())