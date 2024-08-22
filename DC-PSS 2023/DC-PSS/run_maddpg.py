import gym
import ReplayBuffer
from AgentMADDPG import AgentMADDPG
from StockTradingEnv import StockTradingEnv
# 1. 初始化环境
config = {
    "price_array": price_array,  # 提供价格数组
    "tech_array": ... ,   # 提供技术指标数组
    "turbulence_array": ... ,  # 提供湍流数组
    "if_train": True      # 设置为True表示训练模式
}
env = StockTradingEnv(config)

# 2. 初始化MADDPG智能体
n_agents = 2  # 代表参与的智能体数量
agent_maddpg = AgentMADDPG()
agent_maddpg.init(net_dim=256, state_dim=env.state_dim, action_dim=env.action_dim, n_agents=n_agents)

# 设置训练参数
total_episodes = 5000  # 总共的训练轮数
batch_size = 256
repeat_times = 2
soft_update_tau = 0.05

buffer = ReplayBuffer(max_len=100000, state_dim=env.state_dim, action_dim=env.action_dim, if_use_per=False)

# 3. 在环境中训练MADDPG智能体
for episode in range(total_episodes):
    state = env.reset()
    episode_reward, step_num = 0, 0

    while True:
        action = agent_maddpg.select_actions(state)
        next_state, reward, done, _ = env.step(action)

        buffer.append(state, action, reward, next_state, done)
        episode_reward += reward
        step_num += 1
        state = next_state

        if done:
            break

    if len(buffer) > batch_size:  # 当缓冲区有足够的数据时，开始训练智能体
        agent_maddpg.update_net(buffer, batch_size, repeat_times, soft_update_tau)

    if episode % 10 == 0:
        print(f"Episode: {episode}, Reward: {episode_reward}, Steps: {step_num}")

# 4. 测试智能体的性能
config["if_train"] = False
test_env = StockTradingEnv(config)
state = test_env.reset()
total_test_reward = 0

while True:
    action = agent_maddpg.select_actions(state)
    next_state, reward, done, _ = test_env.step(action)
    total_test_reward += reward
    state = next_state
    if done:
        break

print(f"Test Reward: {total_test_reward}")
