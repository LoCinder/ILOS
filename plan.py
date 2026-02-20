import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import random
import math
import numpy as np
import sys

from HPCSimPickJobs import *

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.rcdefaults()
tf.enable_eager_execution()


def action_from_obs(o):
    lst = []
    for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
        if o[i] == 0 and o[i + 1] == 1 and o[i + 2] == 1 and o[i + 3] == 0:
            pass
        elif o[i] == 1 and o[i + 1] == 1 and o[i + 2] == 1 and o[i + 3] == 1:
            pass
        else:
            lst.append((o[i+1],math.floor(i/JOB_FEATURES)))
    min_time = min([i[0] for i in lst])
    result = [i[1] for i in lst if i[0]==min_time]
    return result[0]


#@profile
def run_policy(env, get_probs, get_out, nums, iters, score_type):
    our_r = []
    op_r = []
    rl_r = []
    f1_r = [] 
    sjf_r = []
    wfp_r = []
    uni_r = []
    fcfs_r = []

    nodecoord = []
    tournode = []
    a4tour = []
    
    with open('tour_nodes_cust.json', 'r') as filet:
        mytour = json.load(filet)

    # time_total = 0
    # num_total = 0
    for iter_num in range(0, iters):
        start = iter_num *args.len
        env.reset_for_test(nums,start)

        f1_r.append(sum(env.schedule_curr_sequence_reset(env.f1_score).values()))
        uni_r.append(sum(env.schedule_curr_sequence_reset(env.uni_score).values()))
        wfp_r.append(sum(env.schedule_curr_sequence_reset(env.wfp_score).values()))
        sjf_r.append(sum(env.schedule_curr_sequence_reset(env.sjf_score).values()))
        fcfs_r.append(sum(env.schedule_curr_sequence_reset(env.fcfs_score).values()))
        
        our_r.append(env.schedule_curr_sequence_reset_cust(mytour[iter_num])) #[4,2,3,1]
        op_start_time = time.time()
        op1r, op1t = env.schedule_curr_sequence_reset_op(env.f1_score, mytour[iter_num])
        op_end_time = time.time()
        print(f"TIMEING: {op_end_time - op_start_time:.4f}")
        ###our_r.append(op1r)
        op_r.append(op1r)
        tournode.append(op1t)

        o = env.build_observation()
        tempcoord = []
        for i in range(0, nums * JOB_FEATURES, JOB_FEATURES):
            tempcoord.append([o[i],o[i+1],o[i+2]])
        nodecoord.append(tempcoord)

        print ("schedule: ", end="")
        rl = 0
        total_decisions = 0

        while True:
            count = 0
            skip_ = []
            lst = []
            for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
                if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
                    lst.append(0)
                elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
                    lst.append(0)
                else:
                    count += 1
                    if all(o[i:i + JOB_FEATURES] == [1] * (JOB_FEATURES-1) + [0]):
                        skip_.append(math.floor(i/JOB_FEATURES))
                    lst.append(1)

            a = action_from_obs(o)
            
            if a in skip_:
                print("SKIP" + "(" + str(count) + ")", end="|")
            else:
                print(str(a) + "(" + str(count) + ")", end="|")
            o, r, d, _ = env.step_for_test(a)
            rl += r
            if d:
                print("Sequence Length:", total_decisions)
                break
        rl_r.append(rl)
        print("")


    with open('tour_nodes.json', 'w') as filet:
        json.dump(tournode, filet)
    with open('nodes_coords.json', 'w') as filen:
        json.dump(nodecoord, filen)
    
    # plot
    print(np.mean(fcfs_r))
    print(np.mean(wfp_r))
    print(np.mean(uni_r))
    print(np.mean(sjf_r))
    print(np.mean(f1_r))
    print(np.mean(rl_r))
    print(np.mean(op_r))
    print(np.mean(our_r))
    #
    all_data = []
    all_data.append(fcfs_r)
    all_data.append(wfp_r)
    all_data.append(uni_r)
    all_data.append(sjf_r)
    all_data.append(f1_r)
    all_data.append(rl_r)
    all_data.append(op_r)
    #all_data.append(our_r)
    

    all_medians = []
    for p in all_data:
        all_medians.append(np.median(p))

    # plt.rc("font", size=45)
    # plt.figure(figsize=(12, 7))
    # plt.ylim([0,400])
    plt.rc("font", size=23)
    plt.figure(figsize=(9, 5))
    axes = plt.axes()

    xticks = [y + 1 for y in range(len(all_data))]
    plt.plot(xticks[0:1], all_data[0:1], 'o', color='darkorange')
    plt.plot(xticks[1:2], all_data[1:2], 'o', color='darkorange')
    plt.plot(xticks[2:3], all_data[2:3], 'o', color='darkorange')
    plt.plot(xticks[3:4], all_data[3:4], 'o', color='darkorange')
    plt.plot(xticks[4:5], all_data[4:5], 'o', color='darkorange')
    plt.plot(xticks[5:6], all_data[5:6], 'o', color='darkorange')
    plt.plot(xticks[6:7], all_data[6:7], 'o', color='darkorange')
    #plt.plot(xticks[7:8], all_data[7:8], 'o', color='darkorange')
    
    #plt.figure(figsize=(10,6))
    box = plt.boxplot(all_data, patch_artist=True, showfliers=True, meanline=True, showmeans=True,
                      boxprops=dict(facecolor='orange', alpha=0.3, edgecolor='darkolivegreen', linewidth=1.5),
                      whiskerprops=dict(color='bisque', linewidth=1.5),
                      medianprops=dict(color='burlywood', linewidth=2))
    plt.grid(True, linestyle='--', alpha=0.7)

    #plt.boxplot(all_data, showfliers=True, meanline=True, showmeans=True, medianprops={"linewidth":0},meanprops={"color":"burlywood", "linewidth":2,"linestyle":"solid"})


    #
    mean_line = mlines.Line2D([], [], color='green', linestyle='--', linewidth=2, label='Mean')
    median_line = mlines.Line2D([], [], color='burlywood', linewidth=2, label='Median')
    sample_points = mlines.Line2D([], [], color='darkorange', marker='o', linestyle='None', label='Samples')
    plt.legend(handles=[sample_points, mean_line, median_line], loc='upper right', frameon=True, fontsize=18)

    axes.yaxis.grid(True)
    axes.set_xticks([y + 1 for y in range(len(all_data))])
    xticklabels = ['FCFS', 'WFP', 'UNI', 'SJF', 'F1', 'RLS', 'ILOS', 'OUR']
    # xticklabels = ['FCFS', 'WFP', 'UNI', 'SJF', 'RL']
    plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))],
             xticklabels=xticklabels)
    if score_type == 0:
        plt.ylabel("Average bounded slowdown")
    elif score_type == 1:
        plt.ylabel("Average Job Wait Time (s)")
    elif score_type == 2:
        plt.ylabel("Average turnaround time")
    elif score_type == 3:
        plt.ylabel("Resource utilization")
    else:
        raise NotImplementedError

    # plt.ylabel("Average waiting time (s)")
    plt.xlabel("Scheduling Methods")
    # plt.tick_params(axis='both', which='major', labelsize=40)
    # plt.tick_params(axis='both', which='minor', labelsize=40)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)

    plt.show()
















if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--rlmodel', type=str, default="./data/logs/lublin256-seed0/lublin256-seed0_s0/")
    parser.add_argument('--workload', type=str, default='./data/SDSC-SP2-1998-4.2-cln.swf')
    #PIK-IPLEX-2009-1.swf   lublin_256.swf   ANL-Intrepid-2009-1.swf   lublin_256_new2   SDSC-SP2-1998-4.2-cln.swf   SDSC-BLUE-2000-4.2-cln.swf   CTC-SP2-1996-3.1-cln.swf   HPC2N-2002-2.2-cln.swf
    parser.add_argument('--len', '-l', type=int, default=64)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--iter', '-i', type=int, default=20)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--score_type', type=int, default=1)
    parser.add_argument('--batch_job_slice', type=int, default=0)

    args = parser.parse_args()

    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    
    # initialize the environment from scratch
    env = HPCEnv(shuffle=args.shuffle, backfil=args.backfil, skip=args.skip, job_score_type=args.score_type,
                 batch_job_slice=args.batch_job_slice, build_sjf=False)
    env.my_init(workload_file=workload_file)
    env.seed(args.seed)

    start = time.time()
    run_policy(env, None, None, args.len, args.iter, args.score_type)
    print("elapse: {}".format(time.time()-start))