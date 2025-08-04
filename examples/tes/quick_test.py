#!/usr/bin/env python3
"""
Quick TES Integration Test

This script provides a minimal test to verify the TES integration is working
without requiring Docker or external TES servers.
"""

import os
import sys
import traceback

# Add APPFL to path
sys.path.insert(0, '../../src')

def test_import():
    """Test that TES modules can be imported."""
    print("🔍 Testing TES module imports...")
    
    try:
        from appfl.comm.tes import TESServerCommunicator, TESClientCommunicator
        print("✅ TES communicators imported successfully")
        
        from appfl.config import ServerAgentConfig, ClientAgentConfig
        print("✅ APPFL config classes imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration creation."""
    print("\n🔧 Testing configuration creation...")
    
    try:
        from appfl.config import ServerAgentConfig, ClientAgentConfig
        
        # Create test configuration
        config_dict = {
            'server_configs': {
                'scheduler': 'SyncScheduler',
                'aggregator': 'FedAvgAggregator',
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
        client_config.client_id = "test_client"
        
        print("✅ Configuration objects created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")
        traceback.print_exc()
        return False

def test_communicator_init():
    """Test TES communicator initialization (without connecting)."""
    print("\n🚀 Testing TES communicator initialization...")
    
    try:
        from appfl.comm.tes import TESServerCommunicator
        from appfl.config import ServerAgentConfig, ClientAgentConfig
        
        # Create minimal config
        config_dict = {
            'server_configs': {
                'scheduler': 'SyncScheduler',
                'aggregator': 'FedAvgAggregator',
                'comm_configs': {
                    'tes_configs': {
                        'tes_endpoint': 'http://localhost:8000',
                        'docker_image': 'test-image:latest'
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
        client_configs = [ClientAgentConfig(**config_dict['client_configs'])]
        client_configs[0].client_id = "test_client"
        
        # This should work even without a TES server running
        tes_comm = TESServerCommunicator(
            server_agent_config=server_config,
            client_agent_configs=client_configs
        )
        
        print(f"✅ TES communicator initialized")
        print(f"   Endpoint: {tes_comm.tes_endpoint}")
        print(f"   Docker image: {tes_comm.docker_image}")
        print(f"   Resource requirements: {tes_comm.resource_requirements}")
        
        # Test task creation (offline)
        import pickle
        test_model = {'weights': [1, 2, 3]}
        model_bytes = pickle.dumps(test_model)
        
        tes_task = tes_comm._create_tes_task(
            client_id="test_client",
            task_name="train",
            model_data=model_bytes,
            metadata={'test': True}
        )
        
        print(f"✅ TES task created successfully")
        print(f"   Task name: {tes_task['name']}")
        print(f"   Has inputs: {len(tes_task.get('inputs', [])) > 0}")
        print(f"   Has outputs: {len(tes_task.get('outputs', [])) > 0}")
        print(f"   Has executors: {len(tes_task.get('executors', [])) > 0}")
        
        # Cleanup
        tes_comm.shutdown_all_clients()
        
        return True
        
    except Exception as e:
        print(f"❌ TES communicator initialization failed: {e}")
        traceback.print_exc()
        return False

def test_client_runner():
    """Test the TES client runner script."""
    print("\n🏃 Testing TES client runner...")
    
    try:
        # Test import of client runner
        from appfl.comm.tes.tes_client_communicator import TESClientCommunicator
        print("✅ TES client communicator imported successfully")
        
        # Test client runner script exists
        client_runner_path = "../../src/appfl/run_tes_client.py"
        if os.path.exists(client_runner_path):
            print("✅ TES client runner script exists")
        else:
            print("❌ TES client runner script not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ TES client runner test failed: {e}")
        return False

def main():
    """Run quick tests."""
    print("⚡ APPFL TES Quick Integration Test")
    print("=" * 40)
    print("This test verifies the basic TES integration without external dependencies.")
    print("")
    
    tests = [
        ("Module Imports", test_import),
        ("Configuration Creation", test_configuration),
        ("Communicator Initialization", test_communicator_init),
        ("Client Runner", test_client_runner)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Quick Test Results:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All quick tests passed!")
        print("\n📋 Your TES integration is ready! Next steps:")
        print("   1. Set up a TES server (Funnel, TESK, etc.)")
        print("   2. Run: python test_tes_basic.py")
        print("   3. Build Docker image and run: python test_tes_federated.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check your APPFL installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())