# rl_project

Questions
1. Does reinforcement learning (particularly learning on the reward signals) incur some space overhead on the network relative to supervised learning?
2. Conversely, can a distilled network learn better than an RL approach with the same network size?
3. How do these factors scale?

Steps
1. Train an RL network
2. Distill it to a smaller network with supervised training
3. Compare size / performance tradeoff of distillation
4. Train that smaller network with same RL algorithm and compare performance
