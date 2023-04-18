# GPU_parallelization_vectorization
Exploring the benefits of using vectorized, parallelized implementations in GPU

## ToDo:
- [ ] Notebook to specify the advantages for vectorized computing 
    - Emphasize on the difference between for-loops vs batch processing
    - Need for matmul computation
- [ ] Implement benchmarking script for different networks
    - [ ] Support both CPU and CUDA device targets
    - [ ] Optional using RT Tensor cores
    - [ ] Vision Networks
        - [ ] ResNet
        - [ ] EfficientNet
        - [ ] ConvNext
        - [ ] Transformer based network
    - [ ] NLP networks
- [ ] Create vectorized RL agents
    - [ ] Using Sumo-highway .cfg?