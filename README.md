# rl_project

Questions
1. Does reinforcement learning (particularly learning on the reward signals) incur some space overhead on the network relative to supervised learning?
2. Conversely, can a distilled network learn better than an RL approach with the same network size?
3. How do these factors scale?

Steps
1. Train an RL network
2. Distill it to a smaller network with supervised training
3. Compare size / performance tradeoff of distillation

<!-- 
External Issues:
There was an issue with torch.save()/load() which resulted in loaded models having strange evaluations (particularly repeated mean scores and very round std deviations). This is suspected to be caused because of the macbook pro 'mps' device which is still in beta and known to have related issues. torch.save() first loads to CPU before loading to the original saved device, hence there was no means to bypass this. Using the sb3 load_from_zip which loads straight to a specified device seems to resolve this issue.

Internal Issues:


Ideas:
Take an action after the update instead of before? -->

Limitations:
While the moethod seems to address compounding errors from an initialized state, it's brittle as learning process guides student to optimal policy in every step resulting in little exploration. However, we argue this is ok so long as the student is never initialized in a bad location.

Method does not continually learn and is not expected to do better than the teacher.

Files:
* `pong.py` trains an expert model from scratch using PPO
* `distill.py` uses supervised learning to distill a saved expert model into a new student model
* `run_script.py` executes several distillation runs for comparisons
* `figures/` contains several scripts for graphing figures

Logging:
* All `pong.py` runs log in `pong_run/` and all `distill.py` runs log in `distill_run/`
* When you start a new run, either rename the old run's log folder or delete it; its contents will be overwritten/appended to
* If you rename it, preferably rename it by appending to it (for example, appending a number) since this is covered by the `.gitignore`
