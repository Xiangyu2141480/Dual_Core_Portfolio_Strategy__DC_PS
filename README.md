# DualCore Portfolio Strategy (DC-PS)

The DualCore Portfolio Strategy (DC-PS) is a cutting-edge multi-agent algorithm specifically designed to tackle the challenges of overfitting and to enhance the generalization ability of policies within multidimensional, continuous action spaces.

## Overview

DC-PS integrates two distinct Deep Reinforcement Learning (DRL) agents:

- **DDPG Agent**: Trained using the Deep Deterministic Policy Gradient (DDPG) algorithm.
- **SAC Agent**: Trained using the Soft Actor-Critic (SAC) algorithm.

These agents work collaboratively, interacting with the environment while sharing a common critic network and interaction trajectories. This collaborative approach:

- **Increases Decision-Making Diversity**: By integrating two distinct agents, DC-PS captures a broader spectrum of decision-making logic.
- **Enhances Generalization**: The shared critic network and trajectory pool bolster the strategy's ability to generalize across different market conditions.

## Key Features

- **Multi-Agent Collaboration**: DDPG and SAC agents work together, leveraging their unique strengths.
- **Common Critic Network**: Both agents share a unified critic network, ensuring consistent and stable learning.
- **Shared Interaction Trajectories**: The agents exchange information through shared trajectories, enhancing the overall decision-making process.
- **Robust Generalization**: Designed to perform well in varied market scenarios, minimizing the risk of overfitting.

## Experimental Results

DC-PS has been empirically validated against traditional DRL agents like DDPG and SAC, demonstrating remarkable performance improvements:

- **2021 Performance**: DC-PS achieved an accumulated return at least **15.7% higher** than the best alternative.
- **2023 Performance**: The advantage of DC-PS grew to **275.9%**, underscoring its superior profitability and robustness in out-of-sample scenarios.

## Conclusion

The DualCore Portfolio Strategy (DC-PS) represents a significant advancement in portfolio management strategies, offering a robust, generalizable, and highly profitable solution for dynamic portfolio optimization.
