# rl-env

This repository contains files to train and evaluate an agent on `gym-minigrid` environment. 

TO-DO: add an image for minigrid environment

In the given example code, the agent is trained using feUdal Network. These files can be easily modified to add different RL algorithms.


### Files and Folders 

`/train.py`: script to train agent. 

`/evaluate.py`: script to evaluate model.

After training or evaluation the output gets saved in the `Results/` folder.

`/algos.py`: contains feUdal Network implementation. 

`model/`: contains code to define neural network architectures required for algorithm implementation.

`envs/`: contains code required to customize the environment.  
 
### Example of Use

1. If using docker then download the image and launch the container: 
```
docker build -t rl_env_image .
``` 
This should enter a bash script.

If not using docker then clone the repository and install the required python libraries. `pip3 install -r requirements.txt`

2. Train the agent.

```
python3 train.py
```

3. Evaluate agent's performance.
```
python3 evaluate.py
```


