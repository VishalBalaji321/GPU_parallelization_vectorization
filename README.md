# GPU_parallelization and Vectorization
Exploring the benefits of using vectorized, parallelized implementations
Benchmarking against sequential for-loops vs matmul in numpy, torch (CPU / GPU)

## ToDo:
- [x] Notebook to specify the advantages for vectorized computing 
    - Emphasize on the difference between for-loops vs batch processing
    - Need for matmul computation
    - Support both CPU and CUDA device targets
    - Use AMP (Automatic mixed precision) for calculation
        - CPU Autocast is considerably slower than normal autocast. Possible Reason: The output is bfloat16 and cpu is not optimized for it
        - GPU Autocast is faster for matmul of larger matrices. Smaller matrices are still significantly faster when computed using normal
- [x] Simple Linear Regression (Matmul vs for-loop)
- [ ] Simple CNN (Matmul vs for-loop)
- [ ] Create vectorized RL agents
    - [ ] Q-Learning with OpenAI Gym
