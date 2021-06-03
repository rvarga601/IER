# Reinforcement Learning Based Motion Control of Spherical Robot

The project is aimed at the development of a reinforcement learning algorithm (SARSA) for the position control of a spherical robot. It was implemented in Python using the OpenAI Gym package. The project was developed for the ME41125 Introduction to Engineering Research subject at TU Delft. The main purpose of the GitHub repository is the distribution of the code for the review process.

For specific information about the system and the general structure of the code please refer to the 'varga_5369509.pdf' document.

## Getting Started

### Methodological information

The code is based on the cart-pole example project in the OpenAI Gym environment. The structure of the code remained the same but the functions are modified to solve the current problem.

### Prerequisites

Requirements for the software
- [OpenAI Gym package](https://gym.openai.com/docs/)
- Python 3.6, Numpy, Matplotlib

### Installing

The 'sphericalrobot.py' (Code folder) file should be copied (or cut) next to the OpenAI Gym example projects for example in the 'classic_control' folder. After placing it in the folder the following line should be added to the end of the "__init__.py" file IN THE SAME FOLDER:

    from gym.envs.classic_control.sphericalrobot import SphericalRobotEnv

One folder above the following content should be added to the also called "__init__.py" file:

    register(
        id='SphericalRobot-v0',
        entry_point='gym.envs.classic_control:SphericalRobotEnv',
    )

The 'pickle' file should be placed next to the 'sphero_learn.py' file.

## Testing the setup

Running the 'sphero_learn.py' script it should not produce any error. Missing packages should be installed.

## License

This project is licensed under [MIT](https://choosealicense.com/licenses/mit/) - see the LICENSE.md file for more details.

## Acknowledgments
  - The physical model was derived from the continuous model developed by Tomoki Ohsawa in his [paper](https://arxiv.org/abs/1808.10106).
  - Many thanks to the organizers of the Introduction to Engineering Research subject for the numerous provided materials, advices and templates.
  - Thanks for the OpenAI Gym developers for creating the framework and the developer of the Cart-pole system example project which the code is based on.
