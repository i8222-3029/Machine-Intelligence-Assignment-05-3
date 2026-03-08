#!/usr/bin/env python3
"""Problem 4.3 - Formulating an MDP (3x4 delivery robot grid)."""

from collections import defaultdict


class GridMDP:
    """3x4 stochastic gridworld MDP described in the assignment image."""

    ACTIONS = ("N", "S", "E", "W")
    ACTION_TO_DELTA = {
        "N": (1, 0),
        "S": (-1, 0),
        "E": (0, 1),
        "W": (0, -1),
    }
    PERPENDICULAR = {
        "N": ("W", "E"),
        "S": ("E", "W"),
        "E": ("N", "S"),
        "W": ("S", "N"),
    }

    def __init__(self):
        self.rows = 3
        self.cols = 4
        self.start_state = (1, 1)
        self.wall = (2, 2)
        self.terminals = {(3, 4): 1.0, (2, 4): -1.0}
        self.default_reward = -0.04

        self.states = [
            (r, c)
            for r in range(1, self.rows + 1)
            for c in range(1, self.cols + 1)
            if (r, c) != self.wall
        ]
        self.non_terminal_states = [s for s in self.states if s not in self.terminals]

    def is_inside_grid(self, state):
        row, col = state
        return 1 <= row <= self.rows and 1 <= col <= self.cols

    def is_wall(self, state):
        return state == self.wall

    def is_terminal(self, state):
        return state in self.terminals

    def reward(self, state):
        if state in self.terminals:
            return self.terminals[state]
        return self.default_reward

    def move(self, state, action):
        if self.is_terminal(state):
            return state

        dr, dc = self.ACTION_TO_DELTA[action]
        candidate = (state[0] + dr, state[1] + dc)

        if (not self.is_inside_grid(candidate)) or self.is_wall(candidate):
            return state
        return candidate

    def transition_distribution(self, state, action):
        """Return T(s' | s, a) as dict[state] = probability."""
        if self.is_terminal(state):
            return {state: 1.0}

        distribution = defaultdict(float)

        intended = self.move(state, action)
        distribution[intended] += 0.8

        perp_a, perp_b = self.PERPENDICULAR[action]
        distribution[self.move(state, perp_a)] += 0.1
        distribution[self.move(state, perp_b)] += 0.1

        return dict(sorted(distribution.items()))

    def bellman_q(self, state, action, values, gamma=1.0):
        transitions = self.transition_distribution(state, action)
        q_value = self.reward(state)
        for next_state, prob in transitions.items():
            q_value += prob * (gamma * values[next_state])
        return q_value

    def value_iteration_step(self, values, gamma=1.0):
        new_values = values.copy()

        for state in self.states:
            if self.is_terminal(state):
                new_values[state] = self.reward(state)
                continue

            action_values = [self.bellman_q(state, action, values, gamma) for action in self.ACTIONS]
            new_values[state] = max(action_values)

        return new_values

    def greedy_policy(self, values, gamma=1.0):
        policy = {}
        for state in self.states:
            if self.is_terminal(state):
                policy[state] = "T"
                continue

            best_action = None
            best_q = float("-inf")
            for action in self.ACTIONS:
                q_value = self.bellman_q(state, action, values, gamma)
                if q_value > best_q:
                    best_q = q_value
                    best_action = action

            policy[state] = best_action

        return policy


def fmt_state(state):
    return f"({state[0]},{state[1]})"


def print_value_grid(mdp, values, title):
    print(title)
    for row in range(mdp.rows, 0, -1):
        rendered = []
        for col in range(1, mdp.cols + 1):
            state = (row, col)
            if state == mdp.wall:
                rendered.append("  WALL  ")
            elif state in mdp.terminals:
                rendered.append(f"{values[state]:>7.2f}")
            else:
                rendered.append(f"{values[state]:>7.3f}")
        print(" ".join(rendered))
    print()


def print_policy_grid(mdp, policy, title):
    arrow = {"N": "↑", "S": "↓", "E": "→", "W": "←", "T": "T"}
    print(title)
    for row in range(mdp.rows, 0, -1):
        rendered = []
        for col in range(1, mdp.cols + 1):
            state = (row, col)
            if state == mdp.wall:
                rendered.append("WALL")
            else:
                rendered.append(f"  {arrow[policy[state]]}  ")
        print(" ".join(rendered))
    print()


def print_task_outputs(mdp):
    print("=" * 72)
    print("Problem 4.3 - Formulating an MDP")
    print("=" * 72)
    print()

    print("[1] State and action spaces")
    print(f"Actions A = {mdp.ACTIONS}")
    print("States S (excluding wall):")
    print(", ".join(fmt_state(s) for s in sorted(mdp.states)))
    print(f"Non-terminal states count: {len(mdp.non_terminal_states)}")
    print()

    print("[2] Transition probabilities for s=(1,2), a=East")
    s = (1, 2)
    a = "E"
    distribution = mdp.transition_distribution(s, a)
    for next_state, prob in distribution.items():
        print(f"T({fmt_state(next_state)} | {fmt_state(s)}, {a}) = {prob:.2f}")
    print()

    print("[3] Bellman optimality equation for state (3,2)")
    print("V*(3,2) = max_a [ R(3,2) + γ Σ_{s'} T(s'|(3,2),a) V*(s') ]")
    print("Expanded examples:")

    transitions_n = mdp.transition_distribution((3, 2), "N")
    terms_n = " + ".join(
        [
            f"{p:.1f}V*{fmt_state(ns)}"
            for ns, p in transitions_n.items()
        ]
    )
    print(f"Q*((3,2),N) = R(3,2) + γ[{terms_n}]")

    transitions_e = mdp.transition_distribution((3, 2), "E")
    terms_e = " + ".join(
        [
            f"{p:.1f}V*{fmt_state(ns)}"
            for ns, p in transitions_e.items()
        ]
    )
    print(f"Q*((3,2),E) = R(3,2) + γ[{terms_e}]")
    print()

    print("[4] Value iteration by hand (γ=1)")
    v0 = {state: 0.0 for state in mdp.states}
    for t_state in mdp.terminals:
        v0[t_state] = mdp.reward(t_state)

    v1 = mdp.value_iteration_step(v0, gamma=1.0)
    v2 = mdp.value_iteration_step(v1, gamma=1.0)

    print_value_grid(mdp, v0, "V0")
    print_value_grid(mdp, v1, "V1")
    print(f"V2(3,2) = {v2[(3, 2)]:.4f}")
    print(f"V2(3,1) = {v2[(3, 1)]:.4f}")
    print()

    print("[5] Qualitative policy sketch")
    values = v0
    for _ in range(100):
        values = mdp.value_iteration_step(values, gamma=1.0)

    policy = mdp.greedy_policy(values, gamma=1.0)
    print_policy_grid(mdp, policy, "Greedy policy from converged values")
    print("Caution states (near hazard -1): (2,3), (1,3), (3,3)")
    print("Policy tends to avoid risky cells when stochastic slip can move into (2,4).")


def main():
    mdp = GridMDP()
    print_task_outputs(mdp)


if __name__ == "__main__":
    main()
