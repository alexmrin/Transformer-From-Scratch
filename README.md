# Transformer-From-Scratch

This repository contains the implementation of a Transformer model built entirely from scratch. The purpose of this project is to explore and understand the mechanics behind the Transformer architecture and its modern modifications such as Grouped Query Attention, Rotary Positional Embeddings, and gated ffns.

## Installation
First, install all the required python dependencies.
```bash
pip install -r requirements.txt
```

## Dataset

The model is trained using the OpenWebText dataset. This corpus is an open-source version of the dataset used GPT-2, mirroring the content from WebText data by scraping web pages included in a Reddit dataset.

### Preparing the Data

To download and prepare the OpenWebText dataset, run the following command:

```bash
python -m data.openwebtext.prepare
```

### Data Handling Challenges and Solutions
#### Optimizing Data Loading for Distributed Training
During development, I encountered performance bottlenecks with my data loading strategy, particularly when using PyTorch's Distributed Data Parallel (DDP) for training. The root cause was my use of `np.memmap` within our custom Dataset class, which allows lazy loading of data from disk but does not handle random data access efficiently.

**Problem**: The DDP's default Sampler, which retrieves elements randomly across the entire dataset, interacted poorly with `np.memmap`, leading to significantly slowed training times due to the random disk I/O operations.

**Solution**: To address this, I implemented a custom Sampler class. This sampler allocates a contiguous block of data to each process and samples data sequentially within these blocks. To prevent data leakage and ensure model robustness, each data block is made mutually exclusive by using a sliding window approach, where each example spans a `max_seq_len`. This method significantly reduced disk I/O and improved the overall training speed.

## Training Configuration

### Hardware and Scaling

I trained the model on 8xA100 GPUs with 80GB of memory each. The training lasted for 2 days, utilizing a total batch size of 2240, which was split into micro-batches of size 56 to maximize memory usage, efficiency, and scaling.

### Dropout and Pretraining

Given the vast size of the OpenWebText dataset and the unlikely scenario of encountering many duplicate samples during training, I opted to use a dropout rate of 0 during the pretraining phase. This decision was made to maximize the learning from each unique example and ensure the robustness of the model. I used a warmup with that lasted 7% of the total number of steps and utilized a cosine decay.

### Running the Training

To initiate the training process using the configuration specified, you can use the following `torchrun` command:

```bash
torchrun --nproc_per_node=4 -m scripts.train
```
Change --nproc_per_node depending on how many GPUs you have on your machine.
