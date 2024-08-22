import os
import time
import torch
import numpy as np
from Gym import build_env
from Gym import build_eval_env
from Evaluator import Evaluator
from Replay_buffer import ReplayBuffer

def train_and_evaluate(args, learner_id=0):  # 2021.11.11
    args.init_before_training()  # necessary!

    '''init: Agent'''
    agent = args.agent
    agent.init(net_dim=args.net_dim, state_dim=args.state_dim, action_dim=args.action_dim,
               gamma=args.gamma, reward_scale=args.reward_scale,
               learning_rate=args.learning_rate, if_per_or_gae=args.if_per_or_gae,
               env_num=args.env_num, gpu_id=args.learner_gpus[learner_id], )

    agent.save_or_load_agent(args.cwd, if_save=False)

    env = build_env(env=args.env, if_print=False,
                    env_num=args.env_num, device_id=args.eval_gpu_id, args=args, )
    env_test = build_env(env=args.env_test, if_print=False,
                    env_num=args.env_num, device_id=args.eval_gpu_id, args=args, )
    if env.env_num == 1:
        state, state_EIIE, future_return_based_on_closing = env.reset()
        agent.states = [state_EIIE, ]
        assert isinstance(agent.states[0], np.ndarray)
        assert agent.states[0].shape in {(env.window_size - 5, env.state_dim1 , 3)}
    else:
        # agent.states = env.reset()
        state, state_EIIE, future_return_based_on_closing = env.reset()
        agent.states = [state_EIIE, ]
        assert isinstance(agent.states, torch.Tensor)
        assert agent.states.shape == (env.env_num, env.state_dim)

    '''init Evaluator'''
    eval_env = build_eval_env(args.eval_env, args.env, args.env_num, args.eval_gpu_id, args)
    '''构建测试回测环境'''
    eval_test_env = build_eval_env(args.eval_test_env, args.env_test, args.env_num, args.eval_gpu_id, args)
    evaluator = Evaluator(cwd=args.cwd, agent_id=0,
                          eval_env=eval_env, eval_test_env = eval_test_env,
                          eval_gap=args.eval_gap,
                          eval_times1=args.eval_times1, eval_times2=args.eval_times2,
                          target_return=args.target_return, if_overwrite=args.if_overwrite)
    evaluator.save_or_load_recoder(if_save=False)

    '''init ReplayBuffer'''
    if args.if_off_policy:
        # DDPG属于这部分
        buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_dim,
                              action_dim=1 if env.if_discrete else env.action_dim,
                              if_use_per=args.if_per_or_gae, gpu_id=args.learner_gpus[learner_id])
        buffer.save_or_load_history(args.cwd, if_save=False)

        def update_buffer(_traj_list):
            ten_state, ten_action, ten_state_EIIE, ten_future_return_based_on_closing, ten_other = _traj_list[0]
            buffer.extend_buffer(ten_state, ten_action, ten_state_EIIE, ten_future_return_based_on_closing, ten_other)
            _steps, _r_exp = get_step_r_exp(ten_reward=ten_other[:,0])
            return _steps, _r_exp

        def update_buffer_of_action_and_other(_traj_list):
            ten_action, ten_other = _traj_list[0]
            buffer.extend_buffer_of_actions_others(ten_action, ten_other)
            _steps, _r_exp = get_step_r_exp(ten_reward=ten_other[:,0])
            return _steps, _r_exp

    else:
        buffer = list()
        def update_buffer(_traj_list):
            (ten_state, ten_reward, ten_mask, ten_action, ten_noise) = _traj_list[0]
            buffer[:] = (ten_state.squeeze(1),
                         ten_reward,
                         ten_mask,
                         ten_action.squeeze(1),
                         ten_noise.squeeze(1))

            _step, _r_exp = get_step_r_exp(ten_reward=buffer[1])
            return _step, _r_exp

    """start training"""
    cwd = args.cwd
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    if_allow_break = args.if_allow_break
    soft_update_tau = args.soft_update_tau
    del args

    '''init ReplayBuffer after training start'''
    if agent.if_off_policy:
        if_load = buffer.save_or_load_history(cwd, if_save=False)

        if not if_load:
            traj_list = agent.explore_one_env_for_deep_learning(env, target_step)
            steps, r_exp = update_buffer(traj_list)

    '''start training loop'''
    if_train = True
    while if_train:
        # for i in range(repeat_times):
        #     with torch.no_grad():
        #         traj_list = agent.explore_one_env_for_deep_learning(env, target_step)
        #         buffer.construct_new_buffer()
        #         steps, r_exp = update_buffer(traj_list)
        # logging_tuple = agent.update_net(buffer, batch_size, repeat_times/repeat_times, soft_update_tau)
        with torch.no_grad():
            None
            # traj_list = agent.explore_one_env_for_deep_learning(env, target_step)
            # traj_list = agent.explore_one_env_for_updating_portfolio_weights_vector(env, target_step)
            # buffer.construct_new_buffer_of_action_and_other()
            # steps, r_exp = update_buffer_of_action_and_other(traj_list)
        logging_tuple, action_update, indices = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
        buffer.update_action_buffer(action_update, indices)
        with torch.no_grad():
            steps = repeat_times
            if_reach_goal, if_save = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple)
            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))

    print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if agent.if_off_policy else None
    evaluator.save_or_load_recoder(if_save=True)


def get_step_r_exp(ten_reward):
    return len(ten_reward), ten_reward.sum().item()
    # return len(ten_reward), ten_reward.mean().item()

