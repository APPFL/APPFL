# Instructions: Using Globus Compute for Federated Learning

## FL Client Setup
To setup a **real-world** federated learning client on a **distributed** computing machine, you need to (1) install the APPFL package locally, and (2) create and start a Globus Compute endpoint. [**Note**: Globus Compute endpoint is only supported on linux machine. If you use windows, please use WSL.]

1. Create a virtual environment using `conda`
    ```
    conda create -n APPFL python=3.8
    conda activate APPFL
    ```

2. Install APPFL-Client package.
    ```
    git clone https://github.com/APPFL/APPFLx-Client.git appflx && cd appflx
    pip install -r requirements.txt
    pip install -e .
    ```

3. Creat a Globus account at https://app.globus.org. If you can find your organization in Globus, it is highly recommeneded to use your organization account to log in to Globus as that makes it easier for your collaborators to verify your identity (which is very import for building trust in FL). Otherwise, you can register a Globus account using your commonly-used email address.

4. Set up a Globus Compute endpoint using the following command, where `<YOUR_ENDPOINT_NAME>` is a name of your choice. You might be required to login with Globus for the first time, follow the url and complete all steps to get credential token. 
    ```
    globus-compute-endpoint configure <YOUR_ENDPOINT_NAME>
    ```

5. Configure the endpoint. The command above will create a configuration file at `$HOME/.globus_compute/<YOUR_ENDPOINT_NAME>/config.yaml`. This file should be updated with the appropriate configurations for the computational system you are targeting before you start the endpoint. Globus Compute document shows some of the example setups [here](https://funcx.readthedocs.io/en/latest/endpoints.html#example-configurations).

6. Start the endpoint. Before starting the funcX endpoint, you need to first change to a certain directory you want, which will be the root directory for funcX when accessing your file system or writing output files. Please select that root directory carefully. When you are in your desired directory, run the following command by replacing `<YOUR_ENDPOINT_NAME>` with your endpoint name to start the funcX endpoint. [**Note**: Whenever you start your endpoint, you should start it in the created conda environment.]
    ```
    globus-compute-endpoint start <YOUR_ENDPOINT_NAME>
    ```

7. Finally, the client need to get and note down the endpoint id by running the following command.
    ```
    globus-compute-endpoint list
    ```

## FL Server Setup
To set up a federated learning server using Globus Compute for real-world FL experiments on heterogeneous computing facilities, you need to do the following steps.

1. Creat a Globus account at https://app.globus.org. Same as above, try to find your institution first, or create an account using your commonly-used email address.

2. As Globus Compute has maximum size for the transfered data, APPFL employs S3 bucket to transfer (the relative large) model. Therefore, you need to have an AWS account and create an S3 bucket, and get the access credential file. Set the bucket name (`s3_bucket`) and the path to the credential file (`s3_creds`) accordingly in the [server configuration file](./configs_server/mnist_fedavg.yaml).

3. Setup other configurations for the server in the [server configuration file](./configs_server/mnist_fedavg.yaml), such as:
- `get_model`: function to get the trained model architecture
- `model`: arguments for the model if needed
- `get_loss`: function to get the training loss function
- `val_metric`: function to validate the trained model

4. The server also needs to collect the following information and files from the clients, and put them in the [client configuration file](./configs_client/mnist.yaml).
- `endpoint_id`: Globus Compute Endpoint ID
- `device`: computing device for the client, `cpu` or `cuda`
- `get_data`: data loader function to load the client local dataset. [**Note**: This function will be run on the client computing machine, on the directory where the client starts the Globus Compute endpoint, so the client has to make sure that the data path in the data loader file is correct.]

5. Finally, the server can start the experiment by running the following script in the `examples` folder.
    ```
    python mnist_gc.py --client_config path_to_client_config.yaml --server_config path_to_server_config.yaml
    ```