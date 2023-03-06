# rl-env

This repository contains files to train and evaluate an agent on `gym-minigrid` environment. 

<p align="center">
    <img width="300" src="https://github.com/lcswillems/rl-starter-files/blob/master/README-rsrc/visualize-keycorridor.gif">
</p>

In the given example code, the agent is trained using feUdal Network. These files can be easily modified to add different RL algorithms.


### Files and Folders 

`/train.py`: script to train agent. 

`/evaluate.py`: script to evaluate model.

After training or evaluation the output gets saved in the `Results/` folder.

`/algos.py`: contains feUdal Network implementation. 

`model/`: contains code to define neural network architectures required for algorithm implementation.

`envs/`: contains code required to customize the environment.  
 
### Procedure

1. Clone the repository.
```
git clone https://github.com/ziangqin-stu/impl_data-effiient-hrl.git
```

3. If using docker then build the image and launch the container: 
```
docker build -t rl_env_image .
docker run -it rl_env_image
``` 
This should enter a bash script.

If not using docker then install the required python libraries. `pip3 install -r requirements.txt`

2. Train the agent.

```
python3 train.py
```

3. Evaluate agent's performance.
```
python3 evaluate.py
```

### References

1.MiniGrid Documentation: https://minigrid.farama.org
2. RL Starter Files: https://github.com/lcswillems/rl-starter-files
3. FeUdal Networks for Hierarchical Learning: https://arxiv.org/abs/1703.01161

