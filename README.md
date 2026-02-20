# Enhancing HPC Batch Job Scheduling via Imitation Learning-Based Search

This repo includes the source code and necessary datasets required to implement ILOS for the paper **"Enhancing HPC Batch Job Scheduling via Imitation Learning-Based Search"**, IPDPS' 2026.


If you find it useful, please cite:
```
@inproceedings{zhou2026enhancing,
  title={Enhancing HPC Batch Job Scheduling via Imitation Learning-Based Search},
  author={Zhou, Zechun and Sun, Jingwei and Ye, Mingfei and Sun, Guangzhong},
  booktitle={IEEE International Parallel and Distributed Processing Symposium},
  year={2026}
}
```

## Implementation

- Installation
```sh
git clone https://github.com/LoCinder/ILOS.git
conda create -n ILOS python=3.9.13
conda activate ILOS
pip install torch torchvision
pip install numpy scipy cython tqdm scikit-learn matplotlib seaborn pandas
pip install tensorboard tensorboard_logger
```


- Quick Start
```sh
# Imitation Learning Prioritization
cd pfss
./scripts/test.sh

# Batch Job Scheduling Simulation
cd ..
python plan.py
```

- Complete Workflow

1. Modify the solver settings for a near-exact search in HPCSimPickJobs.py. Perform trace sampling with plan.py to obtain the annotated training set (features: nodes\_coords.json, target: tour\_nodes.json).

2. Navigate to the pfss directory and train the imitation learning module with ./scripts/train.sh using the annotated training set.

3. Modify the solver settings for the refinement search in HPCSimPickJobs.py. Perform trace sampling with plan.py to obtain the test set (features: nodes\_coords.json, target: tour\_nodes.json).

4. Navigate to the pfss directory and obtain the solutions with ./scripts/test.sh for the test set using the trained imitation learning checkpoint.

5. Run plan.py to evaluate ILOS and the baselines across different traces.


## Related Work

The HPC traces involved in our experiments are disclosed in the following archives:

- [Parallel Workloads Archive: Standard Workload Format](https://www.cs.huji.ac.il/labs/parallel/workload/)
- [Chinese Supercomputers Workloads Archive](https://git.ustc.edu.cn/shenyu/CSWA/)


We thank those github source codes for helping to build our own codes:

- [https://github.com/DIR-LAB/deep-batch-scheduler](https://github.com/DIR-LAB/deep-batch-scheduler)
- [https://github.com/lokali/PFSS-IL](https://github.com/lokali/PFSS-IL)
