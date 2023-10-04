# Setup Instructions for Globus Compute

## Client Setup
For a federated learning client, you need to do the following things.
1. Creat a Globus account at https://app.globus.org

2. Create a virtual environment using `conda`
    ```
    conda create -n APPFL python=3.8
    conda activate APPFL
    ```

3. Install APPFL
    ```
    git clone https://github.com/APPFL/APPFL.git
    cd APPFL
    pip install -e .
    ```

4. Set up globus compute endpoint, where `<YOUR_ENDPOINT_NAME>` is a name of your choice. You might be required to login with Globus for the first time, follow the url and complete all steps to get credential token. More details about how to set up the endpoints is shown [here](https://funcx.readthedocs.io/en/latest/endpoints.html).
    ```
    globus-compute-endpoint configure <YOUR_ENDPOINT_NAME>
    globus-compute-ednpoint start <YOUR_ENDPOINT_NAME>
    ```

5. Finally, the client need to get and note down the endpoint id by running the following command.
    ```
    globus-compute-endpoint list
    ```


## Server Setup
As a server, you need to do the following things
1. Creat a Globus account at https://app.globus.org

2. As globus compute also requires to use S3 bucket to transfer model, you need to have an AWS account and create an S3 bucket, and get the access credential file. Set the bucket name and the path to the credential file accordingly in the [server configuration file](./configs_server/mnist_fedavg.yaml).

3. Obtain the client endpoint ids and names, and put them in the in the [client configuration file](./configs_client/mnist.yaml).

4. Finally, the server can start the experiment by running the following script in the `examples` folder.
    ```
    python mnist_gc.py --client_config path_to_client_config.yaml --server_config path_to_server_config.yaml
    ```