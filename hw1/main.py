#!/usr/bin/python3

import numpy as np

# AI 533 - HW1 - Gridworld
# Matthew Pacey

# TODO : no penalty for each movement (YET)

def get_start_state():
    """
    Return starting state for grid world
    :return:
    """
    # example from lecture slides
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
    if (x, y) == (0, 3):
        return 1
    if (x, y) == (1, 3):
        return -1

    # TODO : what happens if the agent cannot move (i.e. edge cell)
    #   does that mean the value is 0 or the value of the current cell?

    cell = 0 # util[x, y]
    if y < len(util[0]) - 1:
        val_rt = util[x, y + 1]
    else:
        val_rt = 0  # cell
    if x > 0:
        val_up = util[x - 1, y]
    else:
        val_up = 0 # cell
    if x < len(util) - 1:
        val_dn = util[x + 1, y]
    else:
        val_dn = 0 # cell
    if y > 0:
        val_lt = util[x, y - 1]
    else:
        val_lt = 0 # cell

    if x == 0 and y == 2:
        pass

    move_rt = 0.8 * (cell + gamma * val_rt) + 0.1 * (cell + gamma * val_up) + 0.1 * (cell + gamma * val_dn)
    move_lt = 0.8 * (cell + gamma * val_lt) + 0.1 * (cell + gamma * val_up) + 0.1 * (cell + gamma * val_dn)
    move_up = 0.8 * (cell + gamma * val_up) + 0.1 * (cell + gamma * val_lt) + 0.1 * (cell + gamma * val_rt)
    move_dn = 0.8 * (cell + gamma * val_dn) + 0.1 * (cell + gamma * val_lt) + 0.1 * (cell + gamma * val_rt)
    return max(move_rt, move_lt, move_up, move_dn)

def value_iteration(mdp, max_error, gamma):
    # initial value of all states
    util = get_start_state()
    # max change of any state in each loop
    max_change = 0

    print(f"init state")
    print(util)

    for t in range(5):
        next_util = np.copy(util)
        if t == 1:
            pass

        for x in range(3):
            for y in range(4):
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
