import argparse
import time
import datetime
import algos
import tensorboardX
import sys

from utils import utils
from model.feudalNet import FeudalNet
from algos.feudalNet import FeudalNetAlgo
from model.hdqn import HDQN
from algos.hdqn import HDQNAlgo


# Parse arguments
parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: feudal | hdqn (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--epochs", type=int, default=5, 
                    help="Number of epochs for training")
parser.add_argument("--num-internal-steps", type=int, default=400, 
                    help="Maximum no of environment steps for training worker")

# Parameters for feUdalNet algorithm
parser.add_argument("--gamma-worker", type=float, default=0.99, 
                    help="Discount factor for worker")
parser.add_argument("--gamma-manager", type=float, default=0.99, 
                    help="Discount factor for manager")
parser.add_argument("--learning-rate", type=float, default=0.0003, 
                    help="Learning rate")
parser.add_argument("--max-grad-norm", type=float, default=0.5, 
                    help="Maximum norm of gradient")
parser.add_argument("--alpha", type=float, default=0.8, 
                    help="Hyperparamter to regulate the influence of intrinsic reward")

parser.add_argument("--recurrence", type=int, default=4, 
                    help="number of time-steps gradient is backpropagated when using LSTM")

# Parameters for hDQN algorithm
parser.add_argument("--decay-high", type=float, default=20000, 
                    help="Decay rate for high level controllers")
parser.add_argument("--decay-low", type=float, default=10000, 
                    help="Decay rate for high level controllers")
parser.add_argument("--lr-high", type=float, default=0.0004, 
                    help="Learning rate for high level controllers")
parser.add_argument("--lr-low", type=float, default=0.00005, 
                    help="Learning rate for high level controllers")


if __name__ == "__main__":
    args = parser.parse_args()

    # Set run dir
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    # txt_logger.info(f"Device: {utils.device}\n")

    # Load environments
    env = utils.make_env(args.env, args.seed)
    observation_space = env.observation_space.shape
    action_space = env.action_space.n
    txt_logger.info("Environments loaded\n")

    # Load training status
    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_episodes": 0, "update": 0}
    txt_logger.info("Training status loaded\n")
    

    # Load model and the algorithm
    if args.algo == 'feudal':
        model = FeudalNet(observation_shape=observation_space, num_outputs=action_space, c=args.recurrence)
        if "model_state" in status:
            model.load_state_dict(status["model_state"])
        # model.to(utils.device)
        
        algo = FeudalNetAlgo(model, env, args.recurrence, args.gamma_manager, args.gamma_worker, 
                                 args.max_grad_norm, args.alpha, args.learning_rate)
    elif args.algo == 'hdqn':
        sub_goal_space = [0, 1, 2]
        sub_goal_space_length = len(sub_goal_space)
        model = HDQN(observation_space, sub_goal_space_length, action_space)
        env = utils.env_with_subgoals(env)
        if "model_state" in status:
            model.load_state_dict(status["model_state"])
        
        high_replay_buffer = utils.ReplayBuffer()
        low_replay_buffer = utils.ReplayBuffer(capacity=20000)
        algo = HDQNAlgo(env=env, model=model, high_replay_buffer=high_replay_buffer,low_replay_buffer=low_replay_buffer,
                        sub_goal_space=sub_goal_space, action_space=action_space,
                        resize_transformer=utils.resize_transform, internal_steps=args.num_internal_steps,
                        lr_high=args.lr_high, lr_low=args.lr_low, epsilon_decay_high=args.decay_high,
                        epsilon_decay_low=args.decay_low)
        
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))
        
    txt_logger.info("Model loaded\n")

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model
    num_episodes = status["num_episodes"]
    update = status["update"]
    start_time = time.time()

    while num_episodes < args.epochs:
        update_start_time = time.time()
        logs1 = algo.collect_experiences(args.num_internal_steps)
        logs2 = algo.update_parameters(num_episodes)
        logs = {**logs1}
        update_end_time = time.time()

        num_episodes += logs["num_episodes"]
        update += 1

        # Print logs
        if update % args.log_interval == 0:            
            duration = int(time.time() - start_time)
            return_per_episode = logs["return_per_episode"]
            num_steps_per_episode = logs["num_steps_per_episode"]

            header = ["update", "epoch", "duration", "returns", "num_steps"]
            data = [update, num_episodes, duration, return_per_episode, num_steps_per_episode]

            txt_logger.info(
                "update {} | epoch {} | duration {} | returns {:.2f} | num_steps {}"
                .format(*data))


            if status["num_episodes"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_episodes)

        # Save status
        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_episodes": num_episodes, "update": update,
                      "model_state": model.state_dict(), "optimizer_state": algo.optimizer.state_dict()}

            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
