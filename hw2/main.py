#!/usr/bin/python3

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random

# AI 533 - HW2 - Gridworld Episodes
# Matthew Pacey

# format the output of numpy array printing
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

# my debug helper: if true, gridworld values printed at each timestep; if false only final values printed
print_intermediate = False

NUM_EPISODES = 30

# TODO: is the printed reward cumulative or singular

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

# counter for number of times a state is visited across all episodes
visited_states = defaultdict(int)
for i in range(16):
    visited_states[f"s{i}"] = 0

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

def move_action(x, y, action):
    """
    Given starting x,y and action, perform action
    Ensure the agent stays on the grid
    :param x:
    :param y:
    :param action:
    :return:
    """
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
def transition(x, y, action):
    """
    Given a state and desired action, probabilistically determine next state.
    The transition function is 80% chance of going in desired direction, with a 10%
    chance of sliding in neighboring direction. For example, if the desired action is up,
    there is an 80% chance of going up, 10% left and 10% right (0% down).
    After calculating the direction, move that way (ensure robot is kept on the board if it runs into an edge).
    :param x:
    :param y:
    :param action:
    :return:
    """
    move = get_move_dir(action)
    next_x, next_y = move_action(x, y, move)
    reward = rewards[x][y]
    return next_x, next_y, reward, move

def run_episode(ep_num):
    x, y = 0, 0
    sequences = []
    total_reward = 0
    while True:
        start_state = convert_xy_to_state_num(x, y)
        visited_states[start_state] += 1

        action = policy[x][y]
        next_x, next_y, reward, move = transition(x, y, action)
        total_reward += reward
        next_state = convert_xy_to_state_num(next_x, next_y)
        sequence = f"({start_state},{move},{reward})"
        sequences.append(sequence)
        x, y = next_x, next_y
        if next_x == 3 and next_y == 3:
            visited_states[next_state] += 1
            total_reward += rewards[x][y]
            reward = rewards[x][y]
            break



    sequences.append(f"({next_state},o,{reward})")
    print(f"Episode {ep_num + 1}: {{{','.join(sequences)}}} (total reward {total_reward})")

def convert_xy_to_state_num(x, y):
    """
    Given an x, y coord, return the state this corresponds to.
    E.g. 0,2 -> s2; 2,3 -> s11
    :param x:
    :param y:
    :return:
    """
    num = x * 4 + y
    return f"s{num}"

def plot_visits():
    """
    Create histogram showing number of times each state was visited
    :return:
    """
    plt.clf()
    plt.title(f"State Visits Using Fixed Policy With {NUM_EPISODES} Episodes")
    plt.bar(list(visited_states.keys()), visited_states.values())
    plt.xlabel("State")
    plt.ylabel("Number of Visits")
    # plt.show()
    fname = "gridworld-hist.png"
    plt.savefig(fname)
    print(f"Histogram saved to {fname}")

def main():
    """
    Main func: run NUM_EPISODES episodes with the fixed policy
    :return:
    """
    for i in range(NUM_EPISODES):
        run_episode(i)
    print("Episode policy/action key:")
    print("\t> = move right")
    print("\tv = move down")
    print("\t< = move left")
    print("\t^ = move up")
    print("\to = remain/goal state")
    plot_visits()
    print("DONE")


if __name__ == '__main__':
    main()
