# GPU_parallelization_vectorization
Exploring the benefits of using vectorized, parallelized implementations in GPU

## ToDo:
- [ ] Notebook to specify the advantages for vectorized computing 
    - Emphasize on the difference between for-loops vs batch processing
    - Need for matmul computation
- [ ] Implement benchmarking script for different networks
    - [ ] Support both CPU and CUDA device targets
    - [ ] Use AMP (Automatic mixed precision) for calculation
            - CPU Autocast is considerably slower than normal autocast. Possible Reason: The output is bfloat16 and cpu is not optimized for it
            - GPU Autocast is faster for matmul of larger matrices. Smaller matrices are still significantly faster when computed using normal
    - [ ] Simple CNN (Matmul vs for-loop)
- [ ] Create vectorized RL agents
    - [ ] Q-Learning with OpenAI Gym