from train import train_all_groups
from evaluate import run_evaluation


# run training for all groups then evaluate and plot results
def main():
    print("starting training for all experiment groups")
    train_all_groups(num_episodes=10000)

    print("\nstarting evaluation and plot generation")
    run_evaluation(num_eval_episodes=100)

    print("\ndone, check plots/ for result visualizations")


if __name__ == '__main__':
    main()
