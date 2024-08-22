import os
import time
import torch
import numpy as np


class Evaluator:  # [ElegantRL.2021.10.13]
    """
    An ``evaluator`` evaluates agent’s performance and saves models.

    :param cwd: directory path to save the model.
    :param agent_id: agent id.
    :param eval_env: environment object for model evaluation.
    :param eval_gap: time gap for periodical evaluation (in seconds).
    :param eval_times1: number of times that get episode return in first.
    :param eval_times2: number of times that get episode return in second.
    :param target_return: target average episodic return.
    :param if_overwrite: save policy networks with different episodic return separately or overwrite.
    """

    def __init__(self, cwd, agent_id, eval_env, eval_test_env, eval_gap, eval_times1, eval_times2, target_return,
                 if_overwrite):
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.recorder_path = f'{cwd}/recorder.npy'

        self.cwd = cwd
        self.agent_id = agent_id
        self.eval_env = eval_env
        self.eval_test_env = eval_test_env
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.if_overwrite = if_overwrite
        self.target_return = target_return

        self.r_max = -np.inf
        self.eval_time = 0
        self.used_time = 0
        self.total_step = 0
        self.start_time = time.time()
        print(f"{'#' * 80}\n"
              f"{'ID':<3}{'Step':>8}{'maxR':>8} |"
              f"{'avgR':>8}{'ex_stdR':>10}{'Return_risk':>15}{'PureR':>10}{'avgS':>7}{'real_stdR':>12} |"
              f"{'avgRt':>8}{'PavgRt':>10}{'Variancet':>10}|"
              f"{'expR':>8}{'objC':>9}{'etc.':>9}")

    def evaluate_and_save(self, act, steps, r_exp, log_tuple) -> (bool, bool):  # 2021-09-09
        """
        Evaluate and save the model.

        :param act: Actor (policy) network.
        :param steps: training steps for last update.
        :param r_exp: mean reward.
        :param log_tuple: log information.
        :return: a boolean for whether terminates the training process and a boolean for whether save the model.
        """
        self.total_step += steps  # update total training steps
        if True:
            '''evaluate first time'''
            rewards_steps_list = [get_episode_return_and_step(self.eval_env, self.eval_test_env, act)
                                  for _ in range(self.eval_times1)]
            r_avg, r_std, s_avg, s_std, return_risk_avg, episode_return_in_test, episode_pure_return_in_test, reward_variance_real_in_test, return_risk_accumlation, episode_return_in_test_20_days, episode_pure_return_in_test_20_days, episode_return_in_test_40_days, episode_pure_return_in_test_40_days, win_day, win_day_in_test, win_days_in_test_20_days, win_days_in_test_40_days, eposide_horizon_reward = self.get_r_avg_std_s_avg_std(
                rewards_steps_list)

            '''evaluate second time'''
            if r_avg > self.r_max:  # evaluate actor twice to save CPU Usage and keep precision
                rewards_steps_list += [get_episode_return_and_step(self.eval_env, self.eval_test_env, act)
                                       for _ in range(self.eval_times2 - self.eval_times1)]
                r_avg, r_std, s_avg, s_std, return_risk_avg, episode_return_in_test, episode_pure_return_in_test, reward_variance_real_in_test, return_risk_accumlation, episode_return_in_test_20_days, episode_pure_return_in_test_20_days, episode_return_in_test_40_days, episode_pure_return_in_test_40_days, win_day, win_day_in_test, win_days_in_test_20_days, win_days_in_test_40_days, eposide_horizon_reward = self.get_r_avg_std_s_avg_std(
                    rewards_steps_list)

            '''save the policy network'''
            if_save = r_avg > self.r_max
            if if_save:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)

                act_name = 'actor' if self.if_overwrite else f'actor.{self.r_max:08.2f}'
                act_path = f"{self.cwd}/{act_name}.pth"
                torch.save(act.state_dict(), act_path)  # save policy network in *.pth

                print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.3f} |")  # save policy and print

            self.recorder.append((self.total_step, r_avg, r_std, return_risk_avg, r_exp, episode_return_in_test,
                                  episode_pure_return_in_test, return_risk_accumlation, episode_return_in_test_20_days,
                                  episode_pure_return_in_test_20_days, episode_return_in_test_40_days,
                                  episode_pure_return_in_test_40_days,
                                  win_day, win_day_in_test, win_days_in_test_20_days, win_days_in_test_40_days, eposide_horizon_reward,  *log_tuple))  # update recorder

            '''print some information to Terminal'''
            if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'ID':<3}{'Step':>8}{'TargetR':>8} |"
                      f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
                      f"{'UsedTime':>8}  ########\n"
                      f"{self.agent_id:<3}{self.total_step:8.2e}{self.target_return:8.2f} |"
                      f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                      f"{self.used_time:>8}  ########")

            print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.3f} |"
                  f" {r_avg:8.3f} {r_std:7.3f}  {r_avg - r_std:7.3f}  {return_risk_avg:7.3f} {s_avg:7.0f}   {s_std:6.6f}   |"
                  f" {episode_return_in_test:8.3f} {episode_return_in_test_40_days:7.3f} {reward_variance_real_in_test:7.6f} |"
                  f"{r_exp:8.3f} {' '.join(f'{n:7.8f}   ' for n in log_tuple)}")
            self.draw_plot()

            if hasattr(self.eval_env, 'curriculum_learning_for_evaluator'):
                self.eval_env.curriculum_learning_for_evaluator(r_avg)
        return if_reach_goal, if_save

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        """
        Compute the average and standard deviation of episodic reward and step.

        :param rewards_steps_list: the trajectory of evaluation.
        :return: average and standard deviation of episodic reward and step.
        """
        rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)
        r_avg, s_avg, risk_avg, variance_avg, return_risk_avg, r_avg_test, pure_r_avg_test, variance_avg_in_test, return_risk_accumlation_mean, episode_r_in_test_20_days_mean, episode_pure_r_in_test_20_days_mean, episode_r_in_test_40_days_mean, episode_pure_r_in_test_40_days_mean, win_day_mean, win_day_in_test_mean, win_days_in_test_20_days_mean, win_days_in_test_40_days_mean, eposide_horizon_reward_mean = rewards_steps_ary.mean(
            axis=0)  # average of episode return and episode step
        r_std, s_std, risk_std, variance_std, return_risk_std, r_std_test, pure_r_std_test, variance_std_in_test, return_risk_accumlation_std, episode_r_in_test_20_days_std, episode_pure_r_in_test_20_days_std, episode_r_in_test_40_days_std, episode_pure_r_in_test_40_days_std, win_day_std, win_day_in_test_std, win_days_in_test_20_days_std, win_days_in_test_40_days_std, eposide_horizon_reward_std = rewards_steps_ary.std(
            axis=0)  # standard dev. of episode return and episode step
        return r_avg, risk_avg, s_avg, variance_avg, return_risk_avg, r_avg_test, pure_r_avg_test, variance_avg_in_test, return_risk_accumlation_mean, episode_r_in_test_20_days_mean, episode_pure_r_in_test_20_days_mean, episode_r_in_test_40_days_mean, episode_pure_r_in_test_40_days_mean, win_day_mean, win_day_in_test_mean, win_days_in_test_20_days_mean, win_days_in_test_40_days_mean, eposide_horizon_reward_mean
        # r_avg, r_std, s_avg, s_std, return_risk_avg
        # ,episode_r_in_test_20_days_mean, episode_pure_r_in_test_20_days_mean, episode_r_in_test_40_days_mean, episode_pure_r_in_test_40_days_mean

    def save_or_load_recoder(self, if_save):
        """
        If ``if_save`` is true, save the recorder. If ``if_save`` is false and recorder exists, load the recorder.

        :param if_save: save or not.
        """
        if if_save:
            np.save(self.recorder_path, self.recorder)
        elif os.path.exists(self.recorder_path):
            recorder = np.load(self.recorder_path)
            self.recorder = [tuple(i) for i in recorder]  # convert numpy to list
            self.total_step = self.recorder[-1][0]

    def draw_plot(self):
        """
        Draw learning curve.

        """
        if len(self.recorder) == 0:
            print("| save_npy_draw_plot() WARNNING: len(self.recorder)==0")
            return None

        np.save(self.recorder_path, self.recorder)

        '''draw plot and save as png'''
        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        save_title = f"step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"

        save_learning_curve(self.recorder, self.cwd, save_title)


def get_episode_return_and_step(env, eval_test_env, act) -> (float, int):  # [ElegantRL.2021.10.13]
    """
    Evaluate the actor (policy) network on testing environment.

    :param env: environment object in ElegantRL.
    :param act: Actor (policy) network.
    :return: episodic reward and number of steps needed.
    """
    device_id = next(act.parameters()).get_device()  # net.parameters() is a python generator.
    device = torch.device('cpu' if device_id == -1 else f'cuda:{device_id}')

    episode_step = 1
    episode_return = 0.0  # sum of rewards in an episode
    eposide_return_variance = 0.0
    return_risk_profit = 0.0
    horizen_reward = 0.0
    win_day = 0
    max_step = env.max_step
    total_asset = 10000000
    # print("max_step:",max_step)
    if_discrete = env.if_discrete
    if if_discrete:
        def get_action(_state):
            _state = torch.as_tensor(_state, dtype=torch.float32, device=device)
            _action = act(_state.unsqueeze(0))
            _action = _action.argmax(dim=1)[0]
            return _action.detach().cpu().numpy()
    else:
        def get_action(_state):
            _state = torch.as_tensor(_state, dtype=torch.float32, device=device)
            _action = act(_state.unsqueeze(0))[0]
            return _action.detach().cpu().numpy()

    state, state_EIIE, future_return_based_on_closing = env.reset()
    last_asset_value = 10000000
    for episode_step in range(max_step):
        action = get_action(state_EIIE)
        # cash = 1-np.mean(action)
        state, state_EIIE, future_return_based_on_closing, reward, done, _ = env.step(action)
        asset_value = getattr(env, 'total_asset', total_asset)
        log_return = np.log2(asset_value/last_asset_value)
        last_asset_value = asset_value
        # print("reward_test:", reward)
        if episode_step < 1:
            reward_list = log_return
        if episode_step >= 1:
            reward_list = np.hstack((reward_list, log_return))
        episode_return += reward
        if log_return >= 0:
            win_day = win_day + 1
        # print("episode_return:",episode_return)
        if done:
            break
    # print("==================1====================")
    # print("episode_return:",episode_return)
    episode_return = getattr(env, 'episode_return', episode_return)
    eposide_return_variance = getattr(env, 'eposide_return_variance', eposide_return_variance)
    eposide_horizon_reward = getattr(env, 'horizen_reward', horizen_reward)
    return_risk_profit = getattr(env, 'return_risk_profit', return_risk_profit)
    # print("episode_return:",episode_return)
    reward_variance_real = np.var(reward_list)
    print("Sharpe ratio train:", (episode_return/len(reward_list))/(reward_variance_real**(1/2)))

    '''在回测集中进行测试 在整个回测集中的表现'''
    eval_test_env.max_step = eval_test_env.price_ary.shape[0] - 1  # 34+61
    max_step_test = eval_test_env.max_step
    state_in_test, state_EIIE_in_test, future_return_based_on_closing_in_test = eval_test_env.reset()
    episode_step_test = 1
    episode_return_in_test = 0
    win_day_in_test = 0
    episode_pure_return_in_test = 0
    last_asset_value = 10000000
    for episode_step_test in range(max_step_test):
        action_in_test = get_action(state_EIIE_in_test)
        state_in_test, state_EIIE_in_test, future_return_based_on_closing_in_test, reward_in_test, done_in_test, _ = eval_test_env.step(action_in_test)
        # print("reward_test:", reward)
        asset_value = getattr(eval_test_env, 'total_asset', total_asset)
        log_return = np.log2(asset_value/last_asset_value)
        last_asset_value = asset_value
        if episode_step_test < 1:
            reward_list_in_test = log_return
        if episode_step_test >= 1:
            reward_list_in_test = np.hstack((reward_list_in_test, log_return))
        if log_return >= 0:
            win_day_in_test = 1 + win_day_in_test
        episode_return_in_test += reward_in_test
        # print("episode_return:",episode_return)
        if done_in_test:
            break
    # print("Date length1:",len(reward_list_in_test))
    episode_return_in_test = getattr(eval_test_env, 'episode_return', episode_return)
    episode_pure_return_in_test = getattr(eval_test_env, 'return_risk_profit', return_risk_profit)
    # eposide_return_variance = getattr(env, 'eposide_return_variance',eposide_return_variance)
    # 累计波动率的计算
    return_risk_accumlation = getattr(eval_test_env, 'eposide_return_variance', eposide_return_variance)
    # print("return_risk_accumlation:",return_risk_accumlation)
    # print("episode_pure_return_in_test_all:",episode_pure_return_in_test)
    # print("episode_return_in_test_all:",episode_return_in_test)
    reward_variance_real_in_test = np.var(reward_list_in_test)
    # print("reward_variance_real:",reward_variance_real)
    print("Sharpe ratio test:", (episode_return_in_test / (len(reward_list_in_test))) / (reward_variance_real_in_test ** (1 / 2)))

    '''在回测集中进行测试 回测期前60天的表现'''
    eval_test_env.max_step = 201 + 60
    max_step_test_2 = 60
    state_in_test_2, state_EIIE_in_test_2, future_return_based_on_closing_in_test_2 = eval_test_env.reset()
    episode_step_test_2 = 1
    episode_return_in_test_20_days = 0
    episode_pure_return_in_test_20_days = 0
    win_days_in_test_20_days = 0
    last_asset_value = 10000000
    for episode_step_test_2 in range(max_step_test_2):
        action_in_test_2 = get_action(state_EIIE_in_test_2)
        state_in_test_2, state_EIIE_in_test_2, future_return_based_on_closing_in_test_2, reward_in_test_2, done_in_test_2, _ = eval_test_env.step(action_in_test_2)
        # print("reward_test:", reward)
        asset_value = getattr(eval_test_env, 'total_asset', total_asset)
        log_return = np.log2(asset_value/last_asset_value)
        last_asset_value = asset_value
        if episode_step_test_2 < 1:
            reward_list_in_test_2 = log_return
        if episode_step_test_2 >= 1:
            reward_list_in_test_2 = np.hstack((reward_list_in_test_2, log_return))
        if log_return>= 0:
            win_days_in_test_20_days = win_days_in_test_20_days + 1
        episode_return_in_test_20_days += reward_in_test_2
        # print("episode_return_20:",episode_return)
        if done_in_test_2:
            break
    # print("Date length2:",len(reward_list_in_test_2))
    episode_return_in_test_20_days = getattr(eval_test_env, 'episode_return', episode_return)
    episode_pure_return_in_test_20_days = getattr(eval_test_env, 'return_risk_profit', return_risk_profit)
    # eposide_return_variance = getattr(env, 'eposide_return_variance',eposide_return_variance)
    # return_risk_profit = getattr(env, 'return_risk_profit', return_risk_profit)
    # print("episode_return_in_test_20:",episode_return_in_test_2)
    # print("episode_pure_return_in_test_20:",episode_pure_return_in_test_2)
    # reward_variance_real_in_test_2 = np.var(reward_list_in_test_2)
    # print("reward_variance_real:",reward_variance_real)
    # episode_pure_return_in_test_20_days = getattr(eval_test_env,'return_without_commission',return_without_commission)
    # print("episode_pure_return_in_test_20_days:",episode_pure_return_in_test_20_days)
    # print("episode_return_in_test_20_days:",episode_return_in_test_20_days)

    '''在回测集中进行测试 回测期前120天的表现'''
    eval_test_env.max_step = 201 + 120
    max_step_test_3 = 121
    state_in_test_3, state_EIIE_in_test_3, future_return_based_on_closing_in_test_3 = eval_test_env.reset()
    episode_step_test_3 = 1
    episode_return_in_test_40_days = 0
    episode_pure_return_in_test_40_days = 0
    win_days_in_test_40_days = 0
    last_asset_value = 10000000
    for episode_step_test_3 in range(max_step_test_3):
        action_in_test_3 = get_action(state_EIIE_in_test_3)
        state_in_test_3, state_EIIE_in_test_3, future_return_based_on_closing_in_test_3, reward_in_test_3, done_in_test_3, _ = eval_test_env.step(action_in_test_3)
        # print("reward_test:", reward)
        asset_value = getattr(eval_test_env, 'total_asset', total_asset)
        log_return = np.log2(asset_value/last_asset_value)
        last_asset_value = asset_value
        if episode_step_test_3 < 1:
            reward_list_in_test_3 = log_return
        if episode_step_test_3 >= 1:
            reward_list_in_test_3 = np.hstack((reward_list_in_test_3, log_return))
        if log_return >= 0:
            win_days_in_test_40_days = win_days_in_test_40_days + 1
        episode_return_in_test_40_days += reward_in_test_3
        # print("episode_return:",episode_return)
        if done_in_test_3:
            break
    # print("Date length3:",len(reward_list_in_test_3))
    episode_return_in_test_40_days = getattr(eval_test_env, 'episode_return', episode_return)
    episode_pure_return_in_test_40_days = getattr(eval_test_env, 'return_risk_profit', return_risk_profit)
    reward_variance_real_in_test_40_days = np.var(reward_list_in_test_3)
    print("Sharpe ratio test 120 days:",
          (episode_return_in_test_40_days / (len(reward_list_in_test_3))) / (
                  reward_variance_real_in_test_40_days ** (1 / 2)))
    # eposide_return_variance = getattr(env, 'eposide_return_variance',eposide_return_variance)
    # return_risk_profit = getattr(env, 'return_risk_profit', return_risk_profit)
    # print("episode_return_in_test_40:",episode_return_in_test_3)
    # print("episode_pure_return_in_test_40:",episode_pure_return_in_test_3)
    # reward_variance_real_in_test_3 = np.var(reward_list_in_test_3)
    # print("reward_variance_real:",reward_variance_real)
    # episode_pure_return_in_test_40_days = getattr(eval_test_env, 'return_without_commission', return_without_commission)
    # print("episode_pure_return_in_test_40_days:",episode_pure_return_in_test_40_days)
    # print("episode_return_in_test_40_days:",episode_return_in_test_40_days)
    # print("================================================================================")
    return episode_return, episode_step, eposide_return_variance, reward_variance_real, return_risk_profit, episode_return_in_test, episode_pure_return_in_test, reward_variance_real_in_test, return_risk_accumlation, episode_return_in_test_20_days, episode_pure_return_in_test_20_days, episode_return_in_test_40_days, episode_pure_return_in_test_40_days, win_day, win_day_in_test, win_days_in_test_20_days, win_days_in_test_40_days, eposide_horizon_reward
    # win_day, win_day_in_test, win_days_in_test_20_days, win_days_in_test_40_days
    # win_day, win_day_in_test, win_days_in_test_20_days, win_days_in_test_40_days
    # episode_return_in_test_20_days, episode_pure_return_in_test_20_days, episode_return_in_test_40_days, episode_pure_return_in_test_40_days


def save_learning_curve(recorder=None, cwd='.', save_title='learning curve', fig_name='plot_learning_curve.jpg'):
    """
    Draw learning curve.

    :param recorder: recorder.
    :param cwd: saving directory.
    :param save_title: learning curve title.
    :param fig_name: figure name.
    """
    if recorder is None:
        recorder = np.load(f"{cwd}/recorder.npy")

    recorder = np.array(recorder)
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    r_avg_pure = recorder[:, 3]
    r_exp = recorder[:, 4]
    episode_return_in_test = recorder[:, 5]
    episode_pure_return_in_test = recorder[:, 6]
    return_risk_accumlation_mean = recorder[:, 7]
    episode_return_in_test_20_days = recorder[:, 8]
    episode_pure_return_in_test_20_days = recorder[:, 9]
    episode_return_in_test_40_days = recorder[:, 10]
    episode_pure_return_in_test_40_days = recorder[:, 11]
    win_day = recorder[:, 12]
    win_day_in_test = recorder[:, 13]
    win_days_in_test_20_days = recorder[:, 14]
    win_days_in_test_40_days = recorder[:, 15]
    eposide_horizon_reward = recorder[:, 16]
    obj_c = recorder[:, 17]
    obj_a = recorder[:, 18]
    # self.total_step, r_avg, r_std, return_risk_avg, r_exp,
    # episode_return_in_test, episode_pure_return_in_test,
    # return_risk_accumlation_mean
    # episode_return_in_test_20_days, episode_pure_return_in_test_20_days,
    # episode_return_in_test_40_days, episode_pure_return_in_test_40_days,
    # *log_tuple

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """
    import matplotlib.pyplot as plt
    # from matplotlib.gridspec import GridSpec
    plt.figure(figsize=(10, 10), dpi=3000)
    # fig = plt.figure(dpi=100,constrained_layout=True) #类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】

    # gs = GridSpec(2, 2, figure=fig) #GridSpec将fiure分为3行3列，每行三个axes，gs为一个matplotlib.gridspec.GridSpec对象，可灵活的切片figure

    fig, axs = plt.subplots(4, 2, figsize=(15, 10), dpi=100)

    '''axs[0, 0]'''
    # ax00 = fig.add_subplot(gs[0:1, 0:1])
    ax00 = axs[0, 0]
    # ax01 = ax00.twinx()
    ax00.cla()

    ax01 = axs[0, 0].twinx()
    color01 = 'darkcyan'
    ax01.set_ylabel('Explore AvgReward', color=color01, fontsize=15)
    # 样本收益（reward的加总：return - modified value * variance）
    ax01.plot(steps, r_exp, color=color01, alpha=0.5, )
    ax01.fill_between(steps, r_exp, 0, facecolor=color01, alpha=0.2, )
    ax01.tick_params(axis='y', labelcolor=color01, labelsize=12)
    ax00.tick_params(axis='x', labelsize=12)

    color00 = 'lightcoral'
    # color001 = 'darkgreen'
    # color002 = 'darkblue'
    ax00.set_ylabel('Episode Return', fontsize=15)
    ax00.tick_params(axis='y', labelsize=12)
    # ax00.plot(steps, r_avg, label='Episode Return', color=color0)
    # ax00.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)
    # ax00.plot(steps, r_avg- r_std, label='Episode Return', color=color00)
    # 纯收益的变化
    ax00.plot(steps, r_avg_pure, label='Episode Return', color=color00)
    # ax00.plot(steps, r_std, color=color001)
    # ax00.plot(steps, r_avg, color=color002)
    # ax00.fill_between(steps, r_avg - r_std, -1, facecolor=color00, alpha=0.2)
    ax00.grid()

    '''axs[1, 0]'''
    # ax10 = fig.add_subplot(gs[0:1, 1:2])
    ax10 = axs[1, 0]
    """
    ax11 = ax10.twinx()    
    """
    ax10.cla()
    color11 = 'darkcyan'
    ax11 = axs[1, 0].twinx()

    ax11.set_ylabel('ObjC', color=color11, fontsize=15)
    ax11.plot(steps, obj_c, color=color11, alpha=0.5, )
    ax11.fill_between(steps, obj_c, facecolor=color11, alpha=0.2, )
    ax11.tick_params(axis='y', labelcolor=color11, labelsize=12)    
    """"""

    ax10.tick_params(axis='x', labelsize=12)

    color10 = 'royalblue'
    # ax10.set_xlabel('Total Steps')
    ax10.set_ylabel('Obj', color=color10, fontsize=15)
    ax10.plot(steps, - obj_a, label='obj', color=color10)
    # ax10.plot(steps, obj_c, color=color11, alpha=0.5, )
    ax10.tick_params(axis='y', labelcolor=color10, labelsize=12)
    """
    for plot_i in range(17, recorder.shape[1]):
        other = recorder[:, plot_i]
        ax10.plot(steps, other, label=f'{plot_i}', color='grey', alpha=0.5)    
    """

    ax10.legend()
    ax10.grid()

    '''axs[2, 0]'''
    # ax20 = fig.add_subplot(gs[1:2, 0:2])
    ax20 = axs[2, 0]
    ax20.cla()

    # ax21 = axs[2].twinx()
    # color01 = 'darkcyan'
    # ax21.set_ylabel('Explore AvgReward', color=color01)
    # ax21.plot(steps, r_exp, color=color01, alpha=0.5, )
    # ax21.tick_params(axis='y', labelcolor=color01)

    color20 = 'lightcoral'
    color201 = 'darkgreen'
    color202 = 'darkblue'
    color203 = 'royalblue'
    # ax20.set_ylabel('Episode Return')
    # ax00.plot(steps, r_avg, label='Episode Return', color=color0)
    # ax00.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)
    # ax20.set_xlabel('Total Steps', fontsize=15)
    ax20.plot(steps, r_avg_pure - r_std, label='Objevtive function', color=color20)
    ax20.plot(steps, r_std, label='Variance', color=color201)
    ax20.plot(steps, r_avg, label='Actual reward', color=color202)
    ax20.plot(steps, r_avg_pure, label='Pure reward', color=color203)
    ax20.fill_between(steps, r_std, 0, facecolor=color201, alpha=0.2)
    ax20.tick_params(axis='y', labelsize=12)
    ax20.tick_params(axis='x', labelsize=12)
    ax20.legend()
    ax20.grid()

    '''axs[3, 0]'''
    # ax30 = fig.add_subplot(gs[1:2, 0:2])
    ax30 = axs[3, 0]
    ax30.cla()
    color30 = 'lightcoral'
    ax30.set_xlabel('Total Steps', fontsize=15)
    ax30.plot(steps, win_day, label='Objevtive function', color=color30)
    ax30.tick_params(axis='y', labelsize=12)
    ax30.tick_params(axis='x', labelsize=12)
    ax30.legend()
    ax30.grid()


    '''axs[0, 1]'''
    # ax20 = fig.add_subplot(gs[1:2, 0:2])
    ax0010 = axs[0, 1]
    ax0010.cla()

    # ax21 = axs[2].twinx()
    # color01 = 'darkcyan'
    # ax21.set_ylabel('Explore AvgReward', color=color01)
    # ax21.plot(steps, r_exp, color=color01, alpha=0.5, )
    # ax21.tick_params(axis='y', labelcolor=color01)

    color0000 = 'lightcoral'
    color0001 = 'darkgreen'
    color0002 = 'darkblue'
    # ax20.set_ylabel('Episode Return')
    # ax00.plot(steps, r_avg, label='Episode Return', color=color0)
    # ax00.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)
    # ax0010.set_xlabel('Total Steps', fontsize = 15)
    # 绘图
    ax0010.plot(steps, episode_return_in_test_20_days, label='Episode return in test 20 days', color=color0000)
    ax0010.plot(steps, episode_return_in_test_40_days, label='Episode return in test 40 days', color=color0001)
    ax0010.plot(steps, episode_return_in_test, label='Episode return in test all days', color=color0002)
    # ax0010.fill_between(steps, r_std, 0, facecolor=color201, alpha=0.2)
    ax0010.tick_params(axis='y', labelsize=12)
    ax0010.tick_params(axis='x', labelsize=12)
    ax0010.legend()
    ax0010.grid()

    '''axs[1, 1]'''
    # ax20 = fig.add_subplot(gs[1:2, 0:2])
    ax0011 = axs[1, 1]
    ax0011.cla()

    # ax21 = axs[2].twinx()
    # color01 = 'darkcyan'
    # ax21.set_ylabel('Explore AvgReward', color=color01)
    # ax21.plot(steps, r_exp, color=color01, alpha=0.5, )
    # ax21.tick_params(axis='y', labelcolor=color01)

    color0010 = 'lightcoral'
    color0011 = 'darkgreen'
    color0012 = 'darkblue'
    # ax20.set_ylabel('Episode Return')
    # ax00.plot(steps, r_avg, label='Episode Return', color=color0)
    # ax00.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)
    # ax0011.set_xlabel('Total Steps', fontsize = 15)
    # 绘图
    ax0011.plot(steps, episode_pure_return_in_test_20_days, label='Episode pure return in test 20 days',
                color=color0010)
    ax0011.plot(steps, episode_pure_return_in_test_40_days, label='Episode pure return in test 40 days',
                color=color0011)
    ax0011.plot(steps, episode_pure_return_in_test, label='Episode pure return in test all days', color=color0012)
    # ax0010.fill_between(steps, r_std, 0, facecolor=color201, alpha=0.2)
    ax0011.tick_params(axis='y', labelsize=12)
    ax0011.tick_params(axis='x', labelsize=12)
    ax0011.legend()
    ax0011.grid()

    '''axs[1, 2]'''
    # ax20 = fig.add_subplot(gs[1:2, 0:2])
    ax0012 = axs[2, 1]
    ax0012.cla()

    # ax21 = axs[2].twinx()
    # color01 = 'darkcyan'
    # ax21.set_ylabel('Explore AvgReward', color=color01)
    # ax21.plot(steps, r_exp, color=color01, alpha=0.5, )
    # ax21.tick_params(axis='y', labelcolor=color01)

    color0020 = 'lightcoral'
    color0021 = 'darkgreen'
    color0022 = 'darkblue'
    color0023 = 'royalblue'
    # ax20.set_ylabel('Episode Return')
    # ax00.plot(steps, r_avg, label='Episode Return', color=color0)
    # ax00.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)
    # ax0012.set_xlabel('Total Steps', fontsize=15)
    # 绘图
    ax0012.plot(steps, episode_pure_return_in_test, label='Pure return in test', color=color0020)
    ax0012.plot(steps, return_risk_accumlation_mean, label='Variance in test', color=color0021)
    ax0012.plot(steps, episode_pure_return_in_test - return_risk_accumlation_mean, label='Objevtive function in test',
                color=color0022)
    ax0012.plot(steps, episode_return_in_test, label='Real reward in test', color=color0023)
    # ax0010.fill_between(steps, r_std, 0, facecolor=color201, alpha=0.2)
    ax0012.tick_params(axis='y', labelsize=12)
    ax0012.tick_params(axis='x', labelsize=12)
    ax0012.legend()
    ax0012.grid()

    '''axs[3, 1]'''
    # ax30 = fig.add_subplot(gs[1:2, 0:2])
    ax0013 = axs[3, 1]
    ax0013.cla()
    color0014 = 'lightcoral'
    color0015 = 'darkgreen'
    color0016 = 'darkblue'
    ax0013.set_xlabel('Total Steps', fontsize=15)
    ax0013.plot(steps, win_day_in_test, label='Objevtive function', color=color0014)
    ax0013.plot(steps, win_days_in_test_20_days, label='Objevtive function', color=color0015)
    ax0013.plot(steps, win_days_in_test_40_days, label='Objevtive function', color=color0016 )
    ax0013.tick_params(axis='y', labelsize=12)
    ax0013.tick_params(axis='x', labelsize=12)
    ax0013.legend()
    ax0013.grid()

    '''plot save'''
    plt.title(save_title, y=13.5, verticalalignment='bottom')
    plt.savefig(f"{cwd}/{fig_name}")
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`



    """The performance of the agent in all the trading period,during the training process"""
    # import matplotlib as mpl
    # mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """
    # import matplotlib.pyplot as plt
    # from matplotlib.gridspec import GridSpec
    plt.figure(figsize=(10, 10), dpi=3000)
    fig_paper, axs_paper = plt.subplots(2, 1, figsize=(15, 10), dpi=100)

    '''axs_paper[0, 0]'''
    # ax00 = fig.add_subplot(gs[0:1, 0:1])
    axs_paper00 = axs_paper[0]
    axs_paper00.cla()

    axs_paper01 = axs_paper[0].twinx()
    color01 = 'darkcyan'
    axs_paper01.set_ylabel('Eposide accumulated return', color=color01, fontsize=20)
    # 样本收益（reward的加总：return - modified value * variance）
    axs_paper01.plot(steps, r_avg, color=color01, alpha=0.5,  label='Eposide accumulated return')
    # axs_paper01.fill_between(steps, r_exp, 0, facecolor=color01, alpha=0.2, )
    axs_paper01.tick_params(axis='y', labelcolor=color01, labelsize=15)
    # axs_paper00.tick_params(axis='x', labelsize=12)

    color00 = 'lightcoral'
    axs_paper00.set_ylabel('Episode horizon reward', fontsize=20)
    axs_paper00.tick_params(axis='y', labelsize=15)
    axs_paper00.tick_params(axis='x', labelsize=15)
    axs_paper00.plot(steps, eposide_horizon_reward, label='Episode horizon reward', color=color00)
    axs_paper00.grid()
    axs_paper00.legend()
    ##############################################################################
    '''axs_paper[1, 0]'''
    axs_paper10 = axs_paper[1]
    axs_paper11 = axs_paper10.twinx()
    axs_paper10.cla()

    # ax11 = axs[1].twinx()
    color11 = 'darkcyan'
    axs_paper11.set_ylabel('Value of the evaluation function', color=color11, fontsize=20)
    axs_paper11.plot(steps, - obj_a, label='Value of the evaluation function', color=color11, alpha=0.5, )
    axs_paper11.tick_params(axis='y', labelcolor=color11, labelsize=15)
    axs_paper10.tick_params(axis='x', labelsize=15)

    color10 = 'royalblue'
    axs_paper10.set_ylabel('Loss function', color=color10, fontsize=20)
    axs_paper10.plot(steps, obj_c, label='Loss function', color=color10)
    axs_paper10.tick_params(axis='y', labelcolor=color10, labelsize=15)
    axs_paper10.legend()
    axs_paper10.grid()


    '''plot save'''
    plt.title("EIIE (Exp.1 (2021))", y=20, verticalalignment='bottom')
    plt.suptitle("EIIE (Exp.1 (2021))", x=0.5, y=0.95, fontsize=25)
    plt.savefig(f"{cwd}/{'plot_learning_curve_convergence.jpg'}")
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    plt.show() # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()