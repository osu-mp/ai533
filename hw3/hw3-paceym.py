#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import random

# AI 533 - HW3 - SARSA and Q-learning
# Matthew Pacey
# TODO : add running instructions

# format the output of numpy array printing
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

# my debug helper: if true, gridworld values printed at each timestep; if false only final values printed
DEBUG = False

gamma = 0.95
NUM_EPISODES = 100
NUM_TRIALS = 100
MAX_STEPS = 1000

actions = ['^', '>', 'v', '<']

"""
This is the policy for the given Gridworld. Starting at the top (0,0) or s0, the ideal
path will be to go down along the left side to the bottom and then right to the goal (3,3) or s15.
"""
policy = np.zeros((4, 4), dtype='U1')
policy = [
    ['v', 'v', '>', 'v'],
    ['v', '>', '>', 'v'],
    ['v', 'v', 'v', 'v'],
    ['>', '>', '>', 'o'],
]

rewards = [
    [-1, -10, -10, -1],
    [-1, -1, -1, -1],
    [-1, -5, -5, -1],
    [-1, -1, -1, 100],
]


def get_init_q():
    """
    Init all state-action pairs to a nominal value
    :return:
    """
    q = {}
    for x in range(4):
        for y in range(4):
            for a in actions:
                q[((x, y), a)] = 0

    return q


def get_move_dir(action):
    """
    Given a desired action, probabilistically determine the actual move.
    The transition function is 80% chance of going in desired direction, with a 10%
    chance of sliding in neighboring direction. For example, if the desired action is up,
    there is an 80% chance of going up, 10% left and 10% right (0% down).
    :param action:
    :return:
    """
    if action == '>':       # move right, can slide up or down
        moves = ['>', '^', 'v']
    elif action == 'v':     # move down, can slide left or right
        moves = ['v', '<', '>']
    elif action == '<':     # move left, can slide up or down
        moves = ['<', '^', 'v']
    if action == '^':       # move up, can slide left or right
        moves = ['^', '<', '>']

    # get a number between 0 and 1
    # if it is less than 0.8, move in primary direction
    # else if less than 0.9, slide in one direction
    # else, slide in other direction
    prob = random.random()
    if prob <= 0.8:
        move = moves[0]
    elif prob <= 0.9:
        move = moves[1]
    else:
        move = moves[2]

    return move


def move_action(s, action):
    """
    Given starting x,y and action, perform action
    Ensure the agent stays on the grid
    :param x:
    :param y:
    :param action:
    :return:
    """
    x, y = s
    if action == '>':
        y = min(y + 1, 3)
    elif action == '<':
        y = max(y - 1, 0)
    elif action == '^':
        x = max(x - 1, 0)
    elif action == 'v':
        x = min(x + 1, 3)
    else:
        raise Exception(f"Invalid action {action}")

    return x, y


def get_epsilon_greedy_action(q, s, epsilon):
    """
    Give q-values and state, determine action using epsilon-greedy policy
    :param q:
    :param s:
    :return:
    """
    # determine whether to use best action or random
    ep_val = random.random()
    if ep_val <= epsilon:           # choose random action
        desired_action = random.choice(actions)
    else:
        desired_action, max_value = get_max_q_value(q, s)

    actual_action = get_move_dir(desired_action)

    if DEBUG:
        print(f"{ep_val=}, {desired_action=}, {actual_action=}")

    return actual_action#, max_value


def get_max_q_value(q, s):
    """
    Given a particular state, return the max q value over all actions
    :param s:
    :return:
    """
    max_value = float('-inf')
    poss_actions = []
    for action in actions:
        value = q[(s, action)]
        if value > max_value:
            poss_actions = [action]
            max_value = value
        elif value == max_value:
            poss_actions.append(action)

    action = random.choice(poss_actions)
    return action, max_value


def SARSA(alpha, epsilon):
    """

    :return:
    """
    all_trials = []

    for trial in range(NUM_TRIALS):
        trial_rewards = []
        q = get_init_q()

        for ep in range(NUM_EPISODES):
            s = (0, 0)
            a = policy[s[0]][s[1]]
            # a = get_epsilon_greedy_action(q, s)
            reward = 0
            for step in range(MAX_STEPS):
                if DEBUG:
                    print(f"{step}: {s=}")
                # take action a, observe R, s'
                #a = get_epsilon_greedy_action(q, s)
                s_prime = move_action(s, a)
                r = rewards[s_prime[0]][s_prime[1]]

                # choose a' from s' using epsilon greedy policy
                a_prime = get_epsilon_greedy_action(q, s_prime, epsilon)
                # r = rewards[s[0]][s[1]]
                reward += r
                q[(s, a)] = q[(s, a)] + alpha * (r + gamma * q[(s_prime, a_prime)] - q[(s, a)])
                s = s_prime
                a = a_prime
                if s == (3, 3):
                    if DEBUG:
                        print(f"GOAL ({step} steps)")
                    # reward += rewards[3][3]
                    trial_rewards.append(reward)
                    break

        all_trials.append(trial_rewards)
    # print(f"SARSA: Avg. reward {np.average(all_rewards)}, stdev {np.std(all_rewards)}")
    return all_trials

def q_learning(alpha, epsilon):
    all_trials = []

    for trial in range(NUM_TRIALS):
        trial_rewards = []
        q = get_init_q()

        for ep in range(NUM_EPISODES):
            s = (0, 0)
            a = policy[s[0]][s[1]]
            reward = 0
            for step in range(MAX_STEPS):
                if DEBUG:
                    print(f"{step}: {s=}")
                # take action a, observe R, s'
                a = get_epsilon_greedy_action(q, s, epsilon)
                s_prime = move_action(s, a)
                r = rewards[s_prime[0]][s_prime[1]]

                # get max reward from s_prime
                desired_action, s_prime_val = get_max_q_value(q, s_prime)
                # r = rewards[s[0]][s[1]]
                reward += r
                q[(s, a)] = q[(s, a)] + alpha * (r + gamma * s_prime_val - q[(s, a)])
                s = s_prime
                if s == (3, 3):
                    if DEBUG:
                        print(f"GOAL ({step} steps)")
                    trial_rewards.append(reward)
                    break

        all_trials.append(trial_rewards)

        # print(f"Q-Learning: Avg. reward {np.average(all_rewards)}, stdev {np.std(all_rewards)}")
    return all_trials


def plot_trials(sarsa, ql):
    plt.clf()
    plt.title("Average Rewards Per Episode")
    plt.xlabel("Episode Number")
    plt.ylabel("Average Reward (with Error Bars)")

    sarsa = np.array(sarsa)
    ql = np.array(ql)

    sarsa_y, sarsa_err = [], []
    ql_y, ql_err = [], []
    print(f"{len(sarsa)},{len(sarsa[0])}")
    for ep in range(NUM_EPISODES):
        sarsa_y.append(np.average(sarsa[:,ep]))
        sarsa_err.append(np.std(sarsa[:,ep]))

        ql_y.append(np.average(ql[:, ep]))
        ql_err.append(np.std(ql[:, ep]))

    # plt.errorbar(list(range(1, NUM_EPISODES + 1)), sarsa_y, sarsa_err, label="SARSA", color='red')
    plt.errorbar(list(range(1, NUM_EPISODES+1)), ql_y, ql_err, label="Q-Learning", color='blue')
    plt.errorbar(list(range(1, NUM_EPISODES+1)), sarsa_y, sarsa_err, label="SARSA", color='red')

    plt.legend(loc='lower right', title='Algorithm')

    fname = "averages.png"
    plt.savefig(fname)
    print(f"Plot saved to {fname}")

def tune_hyperparams():
    """
    Rough way to estimate hyperparams:
    Take average of all trials/episodes across both algos and use the combo with the highest average
    :return:
    """
    best_alpha, best_epsilon = 0, 0
    max_value = float('-inf')

    for alpha in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]:
        for epsilon in [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4]:

            sarsa = np.average(SARSA(alpha, epsilon))
            ql = np.average(q_learning(alpha, epsilon))

            average = (sarsa + ql) / 2
            if DEBUG:
                print(f"{alpha=},{epsilon=} : {average}")
            if average > max_value:
                max_value = average
                best_alpha, best_epsilon = alpha, epsilon


    print(f"Highest average: {best_alpha=},{best_epsilon=},{max_value}")
    return alpha, epsilon


def main():
    alpha, epsilon = tune_hyperparams()

    sarsa_rewards = SARSA(alpha, epsilon)
    ql_rewards = q_learning(alpha, epsilon)
    plot_trials(sarsa_rewards, ql_rewards)


if __name__ == '__main__':
    main()
