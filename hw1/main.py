#!/usr/bin/python3

import numpy as np

# AI 533 - HW1 - Gridworld
# Matthew Pacey

def get_start_state():
    """
    Return starting state for grid world
    :return:
    """
    # example: https://www.youtube.com/watch?v=l87rgLg90HI
    util = np.zeros((3, 4))
    util[0, 3] = 1
    util[1, 3] = -1
    return util
    # gridworld
    util = np.zeros((4, 4))
    util[3, 3] = 100  # goal state, +100 reward
    util[0, 1] = -10  # fire, -10 reward
    util[0, 2] = -10  # fire, -10 reward
    util[2, 1] = -5  # water, -5 reward
    util[2, 2] = -5  # water, -5 reward
    return util


def get_updated_value(x, y, util, gamma):
    """
    Return the max value for a given state by checking movement in all directions
    :param util:
    :param gamma:
    :return:
    """
    cell = util[x, y]
    if y < len(util[0]):
        right_val = util[x, y + 1]
    else:
        right_val = cell
    if x > 0:
        up_val = util[x - 1, y]
    else:
        up_val = cell
    if x < len(util) - 1:
        dn_val = util[x + 1, y]
    else:
        dn_val = cell
    if y > 0:
        left_val = util[x, y - 1]
    else:
        left_val = cell

    right = 0.8 * (cell + gamma * right_val) + 0.1 * (cell + gamma * up_val) + 0.1 * (cell + gamma * dn_val)
    left = 0.8 * (cell + gamma * left_val) + 0.1 * (cell + gamma * up_val) + 0.1 * (cell + gamma * dn_val)
    up = 0.8 * (cell + gamma * up_val) + 0.1 * (cell + gamma * left_val) + 0.1 * (cell + gamma * right_val)
    down = 0.8 * (cell + gamma * dn_val) + 0.1 * (cell + gamma * left_val) + 0.1 * (cell + gamma * right_val)
    return max(right, left, up, down)

def value_iteration(mdp, max_error, gamma):
    # initial value of all states
    util = get_start_state()
    # max change of any state in each loop
    max_change = 0

    for t in range(3):
        next_util = np.copy(util)

        for x in range(3):
            for y in range(4 - 1):      # minus 1 to not calculate for last col
                # print(f"{next_util[x, y]}")
                # move right only now
                # next_util[x, y] = 0.8*(util[x, y] + gamma * util[x, y + 1]) # TODO add 0.1 and 0.1 for other
                next_util[x, y] = get_updated_value(x, y, util, gamma)

        # next_util[0, 2] = 0.72
        # for idx, x in np.ndenumerate(next_util):
        #     next_util[idx] = 0.8*(x + gamma * )
        #     pass
        print(f"time step {t}")
        print(next_util)
        util = next_util

def main():
    value_iteration(None, None, gamma=0.9)
    print("DONE")


if __name__ == '__main__':
    main()
