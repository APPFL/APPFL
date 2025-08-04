#!/usr/bin/env python3
"""
Basic TES Integration Test

This script tests the basic functionality of the APPFL TES integration
without running a full federated learning experiment.
"""

import os
import sys
import json
import time
import requests
import traceback
from omegaconf import OmegaConf

# Add APPFL to path
sys.path.insert(0, '../../src')

def test_tes_endpoint():
    """Test basic TES endpoint connectivity."""
    tes_endpoint = os.getenv('TES_ENDPOINT', 'http://localhost:8000')
    
    print(f"Testing TES endpoint: {tes_endpoint}")
    
    try:
        # Test service info
        response = requests.get(f"{tes_endpoint}/ga4gh/tes/v1/service-info")
        if response.status_code == 200:
            service_info = response.json()
            print(f"✅ TES service info: {service_info.get('name', 'Unknown')}")
            print(f"   Version: {service_info.get('version', 'Unknown')}")
            return True
        else:
            print(f"❌ TES endpoint returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to TES endpoint: {e}")
        return False

def test_tes_task_submission():
    """Test basic TES task submission and monitoring."""
    tes_endpoint = os.getenv('TES_ENDPOINT', 'http://localhost:8000')
    
    print("\nTesting TES task submission...")
    
    # Simple test task
    test_task = {
        "name": "appfl-test-task",
        "description": "APPFL TES integration test",
        "inputs": [],
        "outputs": [{
            "name": "test_output",
            "path": "/tmp/test_output.txt",
            "type": "FILE"
        }],
        "executors": [{
            "image": "python:3.8-slim",
            "command": ["python", "-c", "print('Hello from TES!'); open('/tmp/test_output.txt', 'w').write('Test completed')"],
            "workdir": "/tmp"
        }],
        "resources": {
            "cpu_cores": 1,
            "ram_gb": 1.0,
            "disk_gb": 1.0
        }
    }
    
    try:
        # Submit task
        response = requests.post(
            f"{tes_endpoint}/ga4gh/tes/v1/tasks",
            json=test_task,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"❌ Task submission failed: {response.status_code} - {response.text}")
            return False
        
        task_result = response.json()
        task_id = task_result["id"]
        print(f"✅ Task submitted with ID: {task_id}")
        
        # Monitor task
        print("Monitoring task execution...")
        max_wait = 60  # 1 minute timeout
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            response = requests.get(f"{tes_endpoint}/ga4gh/tes/v1/tasks/{task_id}")
            if response.status_code == 200:
                task_info = response.json()
                state = task_info.get("state", "UNKNOWN")
                print(f"Task state: {state}")
                
                if state == "COMPLETE":
                    print("✅ Task completed successfully!")
                    return True
                elif state in ["SYSTEM_ERROR", "EXECUTOR_ERROR", "CANCELED"]:
                    print(f"❌ Task failed with state: {state}")
                    if "logs" in task_info:
                        print(f"Logs: {task_info['logs']}")
                    return False
                
                time.sleep(2)
            else:
                print(f"❌ Failed to get task status: {response.status_code}")
                return False
        
        print("❌ Task timed out")
        return False
        
    except Exception as e:
        print(f"❌ Task submission error: {e}")
        traceback.print_exc()
        return False

def test_appfl_tes_communicator():
    """Test APPFL TES communicator initialization."""
    print("\nTesting APPFL TES communicator...")
    
    try:
        from appfl.config import ServerAgentConfig, ClientAgentConfig
        from appfl.comm.tes import TESServerCommunicator
        
        # Create minimal config
        config_dict = {
            'server_configs': {
                'scheduler': 'SyncScheduler',
                'aggregator': 'FedAvgAggregator', 
                'num_global_epochs': 1,
                'comm_configs': {
                    'tes_configs': {
                        'tes_endpoint': os.getenv('TES_ENDPOINT', 'http://localhost:8000'),
                        'docker_image': os.getenv('DOCKER_IMAGE', 'python:3.8-slim'),
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
                'train_configs': {
                    'trainer': 'VanillaTrainer',
                    'num_local_steps': 1
                },
                'model_configs': {'model': 'CNN'},
                'data_configs': {'dataset': 'MNIST'}
            }
        }
        
        server_config = ServerAgentConfig(**config_dict)
        client_configs = [ClientAgentConfig(**config_dict['client_configs'])]
        client_configs[0].client_id = "test_client_1"
        
        # Initialize communicator
        tes_comm = TESServerCommunicator(
            server_agent_config=server_config,
            client_agent_configs=client_configs
        )
        
        print(f"✅ TES communicator initialized successfully!")
        print(f"   TES endpoint: {tes_comm.tes_endpoint}")
        print(f"   Docker image: {tes_comm.docker_image}")
        print(f"   Resource requirements: {tes_comm.resource_requirements}")
        
        # Cleanup
        tes_comm.shutdown_all_clients()
        return True
        
    except Exception as e:
        print(f"❌ TES communicator initialization failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all basic tests."""
    print("🧪 APPFL TES Basic Integration Tests")
    print("=" * 50)
    
    tests = [
        ("TES Endpoint Connectivity", test_tes_endpoint),
        ("TES Task Submission", test_tes_task_submission), 
        ("APPFL TES Communicator", test_appfl_tes_communicator)
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
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! TES integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())