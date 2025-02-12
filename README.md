# balance_mjx
Simple MuJoCo mjx code to train a humanoid to balance on one leg

![Humanoid Balancing](https://github.com/ajaytalati/balance_mjx/blob/main/videos/trainned_PPO_policy.gif)

This code adapted from the Playground tutorial - https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb

Also see the linear quadratic regulator code for this task - [MuJoCo humanoid balancing using LQR](https://www.youtube.com/watch?v=RHnXD6uO3Mg) 

# TODO

- Try SAC instead of PPO - the code is in this Colab - https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb
- Try to make the task dynamic, so the humanoid spends 5 seconds balancing on one foot, and then the other, etc... 
