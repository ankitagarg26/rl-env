import minedojo
import argparse

from agent import Agent

# Parse arguments
parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--task-id", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--image-width", default=160
                    help="width of the image")
parser.add_argument("--image-height", default=256
                    help="height of the image")


if __name__ == "__main__":
    args = parser.parse_args()
    
    env = minedojo.make(
        task_id=args.task_id,
        image_size=(args.image_width, args.image_height)
    )
    
    obs = env.reset()
    
    agent = Agent(env)
    agent.learn()
    
    agent.save_results()
    