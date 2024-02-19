FABRIC (FABRIC is Adaptive ProgrammaBle Research Infrastructure for Computer Science and Science Applications) is an International infrastructure that enables cutting-edge experimentation and research at-scale in the areas of networking, cybersecurity, distributed computing, storage, virtual reality, 5G, machine learning, and science applications.

The FABRIC infrastructure is a distributed set of equipment at commercial collocation spaces, national labs and campuses. Each of the 29 FABRIC sites has large amounts of compute and storage, interconnected by high speed, dedicated optical links. It also connects to specialized testbeds (5G/IoT PAWR, NSF Clouds), the Internet and high-performance computing facilities to create a rich environment for a wide variety of experimental activities.

## Configuration

We highly recommend to use [JupyterHub](https://portal.fabric-testbed.net/) to do the configuration.

### Generate Bastion Key

1. Go to [User Profile](https://portal.fabric-testbed.net/user) and click My SSH KEYS on the left column.
2. Click Manage SSH Keys and fill information(Name and Description) to generate Bastion Key Pair.
3. After generation, pls download both your private key and public key. And remember that you cannot download private key again.
4. Move the private key to fabric_config folder in your JupyterHub.

Ref:
1. https://learn.fabric-testbed.net/knowledge-base/logging-into-fabric-vms/

### Set Project ID and generate the configuration
1. Go to JupyterHub and open ```configuration_and_validate.ipynb``` in the example folder.
2. Follow the instruction to generate config using your project id in your user profile.
3. Use the show and validate cell in the jupyter notebook to check if you have set up the config correctly. 


### Use ```APPFL.ipynb``` to setup AFFPL
1. You can find ```APPFL.ipynb``` in our APPFL folder to setup environment.

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
