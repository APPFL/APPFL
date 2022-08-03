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
   ```

   You might be required to login with [Globus](https://app.globus.org/), following the url and complete all steps to get the credential token. 