export N_GPUS=8
export BASE_MODEL='/root/Qwen2.5-7B'
export RM_PATH=/mnt/wx_feature/home/anglv/Think/checkpoints_rm/rm-1.0/checkpoint-84.5

export EXPERIMENT_NAME=self-question-qwen-7b
export VLLM_ATTENTION_BACKEND=XFORMERS

init_train_path="/mnt/wx_feature/home/anglv/verl/datasets/selfq/init_train_200.parquet"
aug_path="/mnt/wx_feature/home/anglv/verl/datasets/selfq/aug.parquet"
math_test_path="/mnt/wx_feature/home/anglv/verl/datasets/selfq/math_test.parquet"
aime_test_path="/mnt/wx_feature/home/anglv/verl/datasets/selfq/aime_test.parquet"
gpqa_test_path="/mnt/wx_feature/home/anglv/verl/datasets/selfq/gpqa_test.parquet"

wandb login 7a8cb30854ae529782b366e9b7164ce7f72632fc

python3 -m verl.trainer.main_ppo \
    data.train_files=[$init_train_path,$aug_path] \
    data.val_files=[$math_test_path,$aime_test_path,$gpqa_test_path] \
    data.data_augment_files=$aug_path \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=6144 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.aug_kwargs.n=4 \
    actor_rollout_ref.rollout.aug_kwargs.top_p=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=0 \
    reward_model.enable=True \
    reward_model.model.path=$RM_PATH \
    reward_model.micro_batch_size=8 \
    critic.optim.lr=5e-6 \
    critic.model.path=$BASE_MODEL \
    critic.ppo_micro_batch_size=8 \
    trainer.logger=['wandb'] \
    trainer.project_name='self_question' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=5 \
    trainer.generate_train_data_freq=5 \
    trainer.critic_warmup=10 \
    trainer.start_to_train_on_aug_data_steps=10 \
    trainer.val_before_train=True \
    trainer.total_training_steps=800 \
    trainer.total_epochs=100000 2>&1 | tee verl_demo_${EXPERIMENT_NAME}.log