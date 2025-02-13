# balance_mjx
Simple MuJoCo mjx code to train a humanoid to balance on one leg

![Humanoid Balancing](https://github.com/ajaytalati/balance_mjx/blob/main/videos/trainned_PPO_policy.gif)

![Humanoid Balancing_front_view](https://github.com/ajaytalati/balance_mjx/blob/main/videos/trainned_PPO_policy_front.gif)

![Humanoid Balancing_overhead_view](https://github.com/ajaytalati/balance_mjx/blob/main/videos/trainned_PPO_policy_overhead.gif)

![Humanoid Balancing_back_view](https://github.com/ajaytalati/balance_mjx/blob/main/videos/trainned_PPO_policy_back_view.gif)

This code is adapted from the Playground tutorial - https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb

Also see the linear quadratic regulator code for this task - [MuJoCo humanoid balancing using LQR](https://www.youtube.com/watch?v=RHnXD6uO3Mg) 

# TODO - Straight line walk balance task

- Try to add stronger penalties/rewards so that the feet are alligned heet-to-toe,
- Instead of focussing on speed, try to make the task dynamic, so the humanoid spends 1 second balancing on one foot, and then the other, etc... 
- Try SAC instead of PPO - the code is in this Colab - https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb

At the moment it runs forward, but wobbles off line ???

![Humanoid Balancing](https://github.com/ajaytalati/balance_mjx/blob/main/videos/straight_line_run_task/trainned_PPO_policy_overhead.gif)

