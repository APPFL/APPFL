# FuncX-APPFL Client Setup

## Prerequisite

 1. Create a Globus account at <https://app.globus.org/>

## APPFL-FuncX Installation

 1. Create a virtual environment (using `conda`):

    ```
    conda create -n appfl python=3.8 pip
    conda activate appfl
    ```

 2. Install APPFL and other required libraries:

    ```
    git clone [https://github.com/APPFL/APPFL.git](https://github.com/APPFL/APPFL.git)
    cd APPFL
    git checkout thoang/funcx
    pip install -r requirements.txt
    pip install -e .
    ```

 3. Install and configure AWS CLI

    ```
    pip install awscli
    aws configure
    ```

    Then use the propvided access key ID and secret key for adding the S3 credential.

 4. Setup [funcx_endpoint](https://funcx.readthedocs.io/en/latest/endpoints.html) client :

   ```
   funcx-endpoint configure <ENDPOINT_NAME>
   funcx-endpoint start <ENDPOINT_NAME>
   ```

   Where `<ENDPOINT_NAME>` is the name of your endpoint, e.g., `uchicago-gpu-01`

   You might be required to login with [Globus](https://app.globus.org/), following the url and complete all steps to get the credential token.

   If your machine is using work management systems, you might need to further configure your endpoint by modifying the `config.py` under `~/.funcx/<ENDPOINT_NAME>/config.py`. For further details, please check this [link](https://funcx.readthedocs.io/en/latest/endpoints.html#example-configurations).

   When ready, you can check the status of your endpoints by running `funcx-endpoint list`, the result should look like this:

   ```
    +--------------------+--------------+--------------------------------------+
    |   Endpoint Name    |    Status    |             Endpoint ID              |
    +====================+==============+======================================+
    | uiuc-cig-01-gpu-02 | Running      | a719450f-4721-4ef1-a5e8-5a22b772d354 |
    +--------------------+--------------+--------------------------------------+
    | uiuc-cig-01-gpu-01 | Running      | 32737688-2eef-4595-a004-3d67992f20a1 |
    +--------------------+--------------+--------------------------------------+
   ```

## Training with your client
   1. Update your client's information under a client config file. Example config files can be found under `configs/clients`.
   2. Update your experiment settings under an experiment config file. Example config files of the `fed_avg` algorithm can be found under `configs/fed_avg`.
   3. Start the APPFL-FuncX federated learning server:
   ```
      python funcx_sync.py \
         --client_config <CLIENT_CONFIG_FILE> \
         --config <EXPERIMENT_CONFIG_FILE>
   ```

## Sharing your client's information
   If you want to join another federated learning experiment, please provide the endpoint information to the federated learning server's owner. Please also make sure that all members need to be under the same Globus Group.
