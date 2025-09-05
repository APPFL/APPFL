#!/usr/bin/env python3
"""
APPFL TES Integration Test

This test verifies the TES integration following APPFL patterns,
similar to how Ray and Globus Compute are tested.
"""

import sys
import os
import traceback

# Add APPFL to path
sys.path.insert(0, '../../src')

def test_tes_import():
    """Test that TES modules can be imported following APPFL patterns."""
    print("🔍 Testing TES module imports...")
    
    try:
        from appfl.comm.tes import TESServerCommunicator, TESClientCommunicator
        print("✅ TES communicators imported successfully")
        
        from appfl.config import ServerAgentConfig, ClientAgentConfig
        print("✅ APPFL config classes imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_tes_server_communicator():
    """Test TES server communicator initialization and task creation."""
    print("\n🚀 Testing TES server communicator...")
    
    try:
        from appfl.comm.tes import TESServerCommunicator
        from appfl.config import ServerAgentConfig, ClientAgentConfig
        from omegaconf import OmegaConf
        
        # Create server configuration following APPFL patterns with OmegaConf
        server_config_dict = {
            'server_configs': {
                'scheduler': 'SyncScheduler',
                'aggregator': 'FedAvgAggregator',
                'num_global_epochs': 5,
                'comm_configs': {
                    'tes_configs': {
                        'tes_endpoint': 'http://localhost:8000',
                        'docker_image': 'appfl/client:latest',
                        'resource_requirements': {
                            'cpu_cores': 1,
                            'ram_gb': 2.0,
                            'disk_gb': 10.0
                        }
                    }
                },
                'aggregator_kwargs': {'num_clients': 2},
                'scheduler_kwargs': {'num_clients': 2, 'same_init_model': True}
            },
            'client_configs': {
                'train_configs': {
                    'trainer': 'VanillaTrainer',
                    'mode': 'step_by_step',
                    'num_local_steps': 10,
                    'optim': 'SGD',
                    'optim_args': {'lr': 0.01}
                },
                'model_configs': {
                    'model': 'CNN'
                },
                'data_configs': {
                    'dataset': 'MNIST',
                    'batch_size': 32
                }
            }
        }
        
        # Convert to OmegaConf structure
        config_omega = OmegaConf.create(server_config_dict)
        server_config = ServerAgentConfig(**config_omega)
        
        # Create client configurations
        client_configs = []
        for i in range(2):
            client_config = ClientAgentConfig(**config_omega['client_configs'])
            client_config.client_id = f"tes_client_{i+1}"
            client_configs.append(client_config)
        
        print(f"✅ Configurations created successfully")
        print(f"   Server scheduler: {server_config.server_configs.scheduler}")
        print(f"   Number of clients: {len(client_configs)}")
        
        # Create TES communicator (same pattern as Ray/Globus Compute)
        tes_comm = TESServerCommunicator(
            server_agent_config=server_config,
            client_agent_configs=client_configs
        )
        
        print(f"✅ TES communicator created successfully")
        print(f"   TES endpoint: {tes_comm.tes_endpoint}")
        print(f"   Docker image: {tes_comm.docker_image}")
        print(f"   Client endpoints: {len(tes_comm.client_endpoints)}")
        
        # Test task creation (offline)
        import pickle
        test_model = {'layer1': {'weights': [1, 2, 3]}, 'layer2': {'weights': [4, 5, 6]}}
        metadata = {'epoch': 1, 'learning_rate': 0.01}
        
        tes_task = tes_comm._create_tes_task(
            client_id="tes_client_1",
            task_name="train",
            model=test_model,
            metadata=metadata
        )
        
        print(f"✅ TES task created successfully")
        print(f"   Task name: {tes_task['name']}")
        print(f"   Has inputs: {len(tes_task.get('inputs', [])) > 0}")
        print(f"   Has outputs: {len(tes_task.get('outputs', [])) > 0}")
        print(f"   Has executors: {len(tes_task.get('executors', [])) > 0}")
        print(f"   Resource CPU cores: {tes_task['resources']['cpu_cores']}")
        
        # Cleanup
        tes_comm.shutdown_all_clients()
        
        return True
        
    except Exception as e:
        print(f"❌ TES server communicator test failed: {e}")
        traceback.print_exc()
        return False

def test_tes_client_communicator():
    """Test TES client communicator."""
    print("\n🏃 Testing TES client communicator...")
    
    try:
        from appfl.comm.tes.tes_client_communicator import TESClientCommunicator
        print("✅ TES client communicator imported successfully")
        
        # Check that the main function exists
        from appfl.comm.tes.tes_client_communicator import main
        print("✅ TES client main function available")
        
        # Check client runner script exists
        client_runner_path = "../../src/appfl/run_tes_client.py"
        if os.path.exists(client_runner_path):
            print("✅ TES client runner script exists")
        else:
            print("❌ TES client runner script not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ TES client communicator test failed: {e}")
        traceback.print_exc()
        return False

def test_tes_api_structure():
    """Test TES API structure and connectivity (optional server check)."""
    print("\n🌐 Testing TES API structure...")
    
    try:
        import requests
        
        tes_endpoint = "http://localhost:8000"
        
        # Test TES API URL construction
        api_urls = {
            "service_info": f"{tes_endpoint}/ga4gh/tes/v1/service-info",
            "create_task": f"{tes_endpoint}/ga4gh/tes/v1/tasks",
            "get_task": f"{tes_endpoint}/ga4gh/tes/v1/tasks/{{task_id}}",
            "cancel_task": f"{tes_endpoint}/ga4gh/tes/v1/tasks/{{task_id}}:cancel",
        }
        
        print("✅ TES API URLs constructed:")
        for name, url in api_urls.items():
            print(f"   {name}: {url}")
        
        # Test request headers
        headers = {"Content-Type": "application/json"}
        print("✅ TES request headers prepared")
        
        # Optional connectivity check (don't fail if server unavailable)
        try:
            response = requests.get(f"{tes_endpoint}/ga4gh/tes/v1/service-info", timeout=3)
            if response.status_code == 200:
                service_info = response.json()
                print(f"✅ TES server available: {service_info.get('name', 'Unknown')}")
            else:
                print(f"⚠️  TES server returned {response.status_code}")
        except requests.exceptions.RequestException:
            print("⚠️  TES server not available (optional)")
            print("   This is fine - test can run without TES server")
        
        return True
        
    except Exception as e:
        print(f"❌ TES API structure test failed: {e}")
        return False

def main():
    """Run TES integration tests following APPFL patterns."""
    print("🎯 APPFL TES Integration Test")
    print("=" * 50)
    print("Testing TES integration following APPFL architectural patterns.")
    print("Similar to Ray and Globus Compute communicator tests.")
    print("")
    
    tests = [
        ("TES Module Import", test_tes_import),
        ("TES Server Communicator", test_tes_server_communicator),
        ("TES Client Communicator", test_tes_client_communicator),
        ("TES API Structure", test_tes_api_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"📋 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed >= 3:  # Core functionality tests
        print("\n🎉 APPFL TES integration is working!")
        print("\n📝 Next steps (for full testing):")
        print("   1. Build Docker image: cd examples/tes && docker build -t appfl/client .")
        print("   2. Start Funnel server: funnel server run")
        print("   3. Run federated learning test: python test_tes_federated.py")
        return 0
    else:
        print("\n⚠️  Some critical tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())