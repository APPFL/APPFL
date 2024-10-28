Active Projects
===============

We currently have the following projects looking for students to work on, especially for those who would like to extend the projects as their master's thesis.

**1. Federated Fine-Tuning of Large Language Models such as LLaMa using Parameter-Efficient Fine-Tuning (PEFT) Methods**

**Abstract:**

Federated learning (FL) is a machine learning paradigm where multiple clients collaboratively train a machine learning model under the orchestration of a central server by sharing the local model trained on the local data. As FL enables training a model by utilizing datasets from multiple clients without explicitly sharing the data, it becomes a promising approach to train a robust and generalized model in scenarios where local datasets cannot be shared due to data privacy, such as in healthcare, smart grid, and insurance domains. Our objective is to fine-tune a Large Language Model (LLM), such as `LLaMa <https://arxiv.org/pdf/2302.13971.pdf>`_, using federated learning techniques. One significant challenge is the large size of LLMs, typically on the order of gigabytes, which makes the process of federated learning both resource-intensive and costly due to the model transfer requirements. To mitigate this, we propose leveraging Parameter-Efficient Fine-Tuning (PEFT) techniques, such as `LoRA <https://arxiv.org/pdf/2106.09685.pdf>`_. PEFT enables efficient adaptation of pre-trained language models to various applications without fine-tuning all the model's parameters. PEFT methods only fine-tune a small number of (extra) model parameters, thereby greatly decreasing the computational and storage costs. In this project, we want to investigate how to utilize PEFT to reduce the transferred model size and finally make federated fine-tuning of LLM feasible and efficient. An alternative method could be `this <https://proceedings.mlr.press/v202/wang23t.html>`_.

**Requirements:**

- Familiar with Python programming and have basic machine learning knowledge.

- Familiar with PyTorch and HuggingFace Transformer libraries is preferred.

**Contact**: Zilinghan Li (zl52@illinois.edu)


**2. Genome-scale Language Model Fine-Tuning using Federated Learning**

**Abstract:**

A research team from Argonne National Laboratory recently developed a `Genome-scale Language Model (GenSLM) <https://www.biorxiv.org/content/biorxiv/early/2022/11/23/2022.10.10.511571.full.pdf>`_ which adapts LLM for genomic data for learning the evolutionary landscape of SARS-CoV-2 genomes. Such a foundation model is pretrained over 110 million prokaryotic gene sequences and fine-tuned on 1.5 million SARS-CoV-2 genomes. We would like to investigate whether we could take the pretrained foundation model and fine-tune the model using federated learning approaches to make the model generalize to other specific downstream tasks.

**Requirements:**

- Familiar with Python programming and have basic machine learning knowledge.

- Familiar with PyTorch and HuggingFace Transformer libraries is preferred.

**Contact**: Zilinghan Li (zl52@illinois.edu), Ravi Madduri (madduri@anl.gov)

**3. Integrating PyTorch RPC and RabbitMQ/MQTT Communication Protocols to the Argonne Privacy-Preserving Federated Learning (APPFL) Framework and More as a Framework Main Developer and Maintainer**

**Abstract:**

`Argonne Privacy-Preserving Federated Learning (APPFL) Framework <https://github.com/APPFL/APPFL>`_ currently supports three communication protocols for federated learning. (1) MPI for FL simulations on one computing machine or HPC cluster. (2) gRPC for running FL on real-world server and client devices. (3) Globus Compute for running FL on real-world server and client devices, especially on HPC and supercomputers. We now would like to integrate PyTorch RPC and RabbitMQ/MQTT protocols to support federated learning for edge devices. Additionally, we want the student to step up to the leadership role to maintain and further develop the APPFL framework.

**Requirements:**

- Familiar with Python programming.

- Familiar with RabbitMQ or MQTT is highly preferred.

- Faimilar with distributed ML is highly preferred.

**Contact**: Zilinghan Li (zl52@illinois.edu)

