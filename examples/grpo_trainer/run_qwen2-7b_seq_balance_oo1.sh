{
    source /map-vepfs/miniconda3/etc/profile.d/conda.sh
    conda activate verl_yizhi

    set -x

    # source /map-vepfs/miniconda3/etc/profile.d/conda.sh
    # conda activate verl_yizhi
    # cd /map-vepfs/yizhi/verl
    # ray start --head
    # bash examples/grpo_trainer/run_qwen2-7b_seq_balance_oo1.sh

    # export VLLM_ATTENTION_BACKEND=XFORMERS

    cp /map-vepfs/yizhi/.netrc /root/.netrc

    worker_num=$MLP_WORKER_NUM # 4
    num_gpu_per_worker=$MLP_WORKER_GPU # 8
    worker_id=$MLP_ROLE_INDEX
    main_woker_ip=$MLP_WORKER_0_HOST
    main_woker_port=$MLP_WORKER_0_PORT


    total_num_gpu=$((${worker_num} * ${num_gpu_per_worker}))

    RAY_CLUSTER_ADDRESS=${main_woker_ip}:6379
    MIN_NUM_WORKERS=${worker_num} # Minimum number of workers required

    # Function to check cluster status
    check_cluster_status() {
        echo `ray status --address $RAY_CLUSTER_ADDRESS`
        ray status --address $RAY_CLUSTER_ADDRESS | grep -q "0.0/${total_num_gpu}.0 GPU"
    }

    main_woker_id=0
    if [[ ${main_woker_id} -eq $worker_id ]]; then
        ray start --head --node-ip-address ${main_woker_ip} --port 6379 --num-gpus ${num_gpu_per_worker}
        # Wait for sufficient workers
        while ! check_cluster_status; do
            echo "Waiting for sufficient workers..."
            sleep 2  # Adjust the sleep interval as needed
        done
        echo "Sufficient workers available. Submitting job..." 
    else
        sleep 20s # wait for head node
        ray start --address ${main_woker_ip}:6379 --num-gpus ${num_gpu_per_worker} --block
    fi


    PROJECT_DIR="/map-vepfs/yizhi/verl"
    cd $PROJECT_DIR

    if [[ ${main_woker_id} -eq $worker_id ]]; then
        python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=grpo \
            data.train_files=${PROJECT_DIR}/data/oo1_rl/sample.parquet \
            data.val_files=${PROJECT_DIR}/data/oo1_rl/sample.parquet \
            data.train_batch_size=1024 \
            data.max_prompt_length=512 \
            data.max_response_length=1024 \
            data.filter_overlong_prompts=True \
            data.truncation='error' \
            actor_rollout_ref.model.path=/map-vepfs/huggingface/models/Qwen/Qwen2.5-Math-7B-128k \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=256 \
            actor_rollout_ref.actor.use_dynamic_bsz=True \
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=0.00001 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
            actor_rollout_ref.rollout.n=32 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            reward_model.reward_manager="naive" \
            algorithm.kl_ctrl.kl_coef=0.00001 \
            trainer.critic_warmup=0 \
            trainer.logger=['console','wandb'] \
            trainer.project_name='verl_grpo_oo1_rl_v5_debug' \
            trainer.experiment_name='qwen2_7b_math_naive-manager_tbsz-1024_Bo32' \
            +trainer.val_before_train=False \
            trainer.n_gpus_per_node=8 \
            trainer.nnodes=2 \
            trainer.save_freq=-1 \
            trainer.test_freq=-1 \
            trainer.total_epochs=15 $@
    fi
}