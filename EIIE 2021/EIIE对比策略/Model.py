import torch
import numpy as np
from AgentDDPG import AgentDDPG
from Config import Arguments
from Run_tutorial import train_and_evaluate

# MODELS = {"ddpg": AgentDDPG, "td3": AgentTD3, "sac": AgentSAC, "ppo": AgentPPO, "a2c": AgentA2C}
MODELS = {"ddpg": AgentDDPG}
OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo", "a2c"]
"""MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}"""


class DRLAgent:
    """Implementations of DRL algorithms
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env, price_array, tech_array, turbulence_array, OCLC_array, close_in_OCLC_array,  price_array_test, tech_array_test,
                 turbulence_array_test, OCLC_array_test, close_in_OCLC_array_test):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
        self.turbulence_array = turbulence_array
        self.OCLC_array = OCLC_array
        self.close_in_OCLC_array = close_in_OCLC_array
        self.price_array_test = price_array_test
        self.tech_array_test = tech_array_test
        self.turbulence_array_test = turbulence_array_test
        self.OCLC_array_test = OCLC_array_test
        self.close_in_OCLC_array_test = close_in_OCLC_array_test

    def get_model(self, model_name, model_kwargs):
        env_config = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "turbulence_array": self.turbulence_array,
            "OCLC_array": self.OCLC_array,
            "close_in_OCLC_array": self.close_in_OCLC_array,
            "if_train": True,
        }
        env_test_config = {"price_array": self.price_array_test,
                           "tech_array": self.tech_array_test,
                           "turbulence_array": self.turbulence_array_test,
                           "OCLC_array": self.OCLC_array_test,
                           "close_in_OCLC_array": self.close_in_OCLC_array_test,
                           "if_train": True}
        env = self.env(config=env_config)
        env_test = self.env(config=env_test_config)
        env.env_num = 1
        agent = MODELS[model_name]()
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        '''将测试机放入模型中'''
        model = Arguments(env, env_test, agent)
        if model_name in OFF_POLICY_MODELS:
            model.if_off_policy = True
        else:
            model.if_off_policy = False

        if model_kwargs is not None:
            try:
                model.learning_rate = model_kwargs["learning_rate"]
                model.batch_size = model_kwargs["batch_size"]
                model.gamma = model_kwargs["gamma"]
                model.seed = model_kwargs["seed"]
                model.random_seed = model_kwargs["seed"]
                model.net_dim = model_kwargs["net_dimension"]
                model.target_step = model_kwargs["target_step"]
                model.eval_gap = model_kwargs["eval_gap"]
                print("random_seed:", model.random_seed)
                print("seed", model.seed)
            except BaseException:
                raise ValueError(
                    "Fail to read arguments, please check 'model_kwargs' input."
                )
        return model

    def train_model(self, model, cwd, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = total_timesteps
        train_and_evaluate(model)

    @staticmethod
    def DRL_prediction(model_name, cwd, net_dimension, environment, environment_add):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        model = MODELS[model_name]()
        environment.env_num = 1
        args_basic = Arguments(env=environment, env_test=environment_add, agent=model)
        if model_name in OFF_POLICY_MODELS:
            args_basic.if_off_policy = True
        else:
            args_basic.if_off_policy = False
        args_basic.agent = model
        args_basic.env = environment

        # args.agent.if_use_cri_target = True  ##Not needed for test

        # load agent
        try:
            state_dim = environment.state_dim
            action_dim = environment.action_dim

            agent = args_basic.agent
            net_dim = net_dimension

            agent.init(net_dim, state_dim, action_dim)
            agent.save_or_load_agent(cwd=cwd, if_save=False)


            act = agent.act
            device = agent.device

        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        _torch = torch
        state, state_EIIE, future_return_based_on_closing = environment.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        episode_total_assets = list()
        episode_returns_modified_by_risk = list()
        episode_returns.append(0)
        episode_returns_modified_by_risk.append(0)
        episode_total_assets.append(environment.initial_total_asset)
        with _torch.no_grad():
            for i in range(environment.max_step):
                s_tensor = _torch.as_tensor((state_EIIE,), device=device)
                # a_tensor = act(s_tensor.to(device))
                # s_tensor.to(torch.float32)
                if i%1 == 0:
                    a_tensor_basic = act(s_tensor.to(torch.float32))  # action_tanh = act.forward()
                    a_tensor = a_tensor_basic # actM(s_tensor.to(torch.float32), a_tensor_basic)
                    action = (a_tensor.detach().cpu().numpy()[0])
                    # print("action1", action)
                    state, state_EIIE, future_return_based_on_closing, reward, done, _ = environment.DRL_agent_step(action,  modified=True)
                    last_action = environment.presentaion_weights
                else:
                    action = last_action
                    # print("action2", action)
                    state, state_EIIE, future_return_based_on_closing, reward, done, _ = environment.DRL_agent_step(action, modified=False)
                    last_action = environment.presentaion_weights

                ### a_tensor = act(s_tensor.to(torch.float32))  # action_tanh = act.forward()
                ### action = (
                ###     a_tensor.detach().cpu().numpy()[0]
                ### )  # not need detach(), because with torch.no_grad() outside
                ### state, reward, done, _ = environment.step(action)

                total_asset = (
                        environment.amount
                        + (
                                environment.price_ary[environment.day] * environment.stocks
                        ).sum()
                )
                # print("total_asset:", total_asset)

                ###
                episode_total_assets.append(total_asset)
                episode_return = np.log(total_asset / environment.initial_total_asset)
                ###
                episode_returns.append(episode_return)
                # print("episode_return",episode_return)
                # self.return_risk_profit
                # return_modified_risk = environment.return_modified_by_risk [0,0]
                # return_modified_risk = environment.return_risk_profit
                return_modified_risk = episode_return - (1/50) * environment.episode_expected_risk
                # return_modified_risk = episode_return - environment.episode_expected_risk
                return_without_commission = environment.return_without_commission
                ###
                episode_returns_modified_by_risk.append(return_modified_risk)

                # self.episode_return - self.eposide_return_variance
                if done:
                    break

        #############################################################################################
        _torch = torch
        state, state_EIIE, future_return_based_on_closing = environment.reset()
        episode_returns_melt = list()  # the cumulative_return / initial_account
        episode_total_assets_melt  = list()
        episode_returns_modified_by_risk_melt  = list()
        ##########################################################################################
        episode_returns_melt.append(0)
        episode_returns_modified_by_risk_melt .append(0)
        episode_total_assets_melt .append(environment.initial_total_asset)
        with _torch.no_grad():
            for i in range(environment.max_step):
                s_tensor = _torch.as_tensor((state_EIIE,), device=device)
                # a_tensor = act(s_tensor.to(device))
                # s_tensor.to(torch.float32)
                if i%1 == 0:
                    a_tensor_basic = act(s_tensor.to(torch.float32))  # action_tanh = act.forward()
                    action_basic = (a_tensor_basic.detach().cpu().numpy()[0])
                    # print("action1", action)
                    state, state_EIIE, future_return_based_on_closing, reward, done, _ = environment.DRL_agent_step(action_basic,  modified=True)
                    last_action_basic = environment.presentaion_weights
                else:
                    action_basic = last_action_basic
                    # print("action2", action)
                    state, state_EIIE, future_return_based_on_closing, reward, done, _ = environment.DRL_agent_step(action_basic, modified=False)
                    last_action_basic = environment.presentaion_weights

                ### a_tensor = act(s_tensor.to(torch.float32))  # action_tanh = act.forward()
                ### action = (
                ###     a_tensor.detach().cpu().numpy()[0]
                ### )  # not need detach(), because with torch.no_grad() outside
                ### state, reward, done, _ = environment.step(action)

                total_asset_melt = (
                        environment.amount
                        + (
                                environment.price_ary[environment.day] * environment.stocks
                        ).sum()
                )
                # print("total_asset:", total_asset)

                ###
                episode_total_assets_melt.append(total_asset_melt)

                episode_return_melt = np.log(total_asset_melt / environment.initial_total_asset)
                ###
                episode_returns_melt.append(episode_return_melt)
                # print("episode_return",episode_return_melt)

                # self.return_risk_profit
                # return_modified_risk = environment.return_modified_by_risk [0,0]
                # return_modified_risk = environment.return_risk_profit
                return_modified_risk_melt = episode_return_melt - (1/50) * environment.episode_expected_risk
                # return_modified_risk = episode_return - environment.episode_expected_risk
                return_without_commission_melt = environment.return_without_commission
                ###
                episode_returns_modified_by_risk_melt.append(return_modified_risk_melt)

                # self.episode_return - self.eposide_return_variance
                if done:
                    break

        print("Test Finished!")
        # return episode total_assets on testing data
        print("episode_return_melt", episode_return_melt)
        print("return_without_commission_melt", return_without_commission_melt)
        return episode_total_assets, episode_returns, episode_returns_modified_by_risk, episode_total_assets_melt, episode_returns_melt, episode_returns_modified_by_risk_melt