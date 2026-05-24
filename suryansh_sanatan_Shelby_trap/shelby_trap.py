import numpy as np


class TwoStepMDP:
    REWARD_PROBS = {1: 0.8, 2: 0.2}

    def step(self, state, action):
        if state == 0:
            p = np.random.random()
            if action == 0:
                next_state = 1 if p < 0.8 else 2
            else:
                next_state = 2 if p < 0.8 else 1
            return next_state, 0.0, False
        elif state in (1, 2):
            reward = 1.0 if np.random.random() < self.REWARD_PROBS[state] else 0.0
            return None, reward, True
        else:
            raise ValueError(f"Unknown state: {state}")


class QLearningAgent:
    def __init__(self, n_actions=2, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = np.zeros((1, n_actions))
        self.V = np.zeros(2)
        self.N_Q = np.zeros((1, n_actions), dtype=int)
        self.N_V = np.zeros(2, dtype=int)

    def choose_action(self, state):
        if state == 0:
            if np.random.random() < self.epsilon:
                return np.random.randint(self.Q.shape[1])
            return int(np.argmax(self.Q[0]))
        return 0

    def update(self, state, action, reward, next_state):
        if state in (1, 2):
            idx = state - 1
            self.V[idx] += self.alpha * (reward - self.V[idx])
            self.N_V[idx] += 1
        else:
            max_next = float(self.V[next_state - 1]) if next_state is not None else 0.0
            td_error = reward + self.gamma * max_next - self.Q[0, action]
            self.Q[0, action] += self.alpha * td_error
            self.N_Q[0, action] += 1

    def print_q_table(self, title="Q-table"):
        print(f"\n{'='*55}")
        print(f"  {title}")
        print(f"{'='*55}")
        print(f"  {'State':<18} {'Action':<8} {'Value':>10}  {'Visits':>7}")
        print(f"  {'-'*45}")
        for a in range(self.Q.shape[1]):
            marker = " <-- best" if a == int(np.argmax(self.Q[0])) else ""
            print(f"  {'S0 (Hub)':<18} {f'A{a+1}':<8} {self.Q[0,a]:>10.6f}  {self.N_Q[0,a]:>7}{marker}")
        for i, name in enumerate(["S1 (Terminal)", "S2 (Terminal)"]):
            print(f"  {name:<18} {'—':<8} {self.V[i]:>10.6f}  {self.N_V[i]:>7}")
        print(f"{'='*55}")


def run_episode(env, agent):
    state = 0
    action0 = agent.choose_action(state)
    next_state, _, _ = env.step(state, action0)
    _, reward, _ = env.step(next_state, 0)
    agent.update(next_state, 0, reward, None)
    agent.update(state, action0, reward, next_state)
    return reward


def train(env, agent, n_episodes=1000, print_every=200):
    total_rewards = []
    print(f"\nTraining for {n_episodes} episodes (α={agent.alpha}, γ={agent.gamma}, ε={agent.epsilon})")
    for ep in range(1, n_episodes + 1):
        r = run_episode(env, agent)
        total_rewards.append(r)
        if ep % print_every == 0:
            avg = sum(total_rewards[-print_every:]) / print_every
            policy = "A1" if np.argmax(agent.Q[0]) == 0 else "A2"
            print(f"  Episode {ep:>5}: avg reward (last {print_every}) = {avg:.3f}  |  policy at S0 = {policy}")
    return total_rewards


def behavioral_probe(agent):
    print("\n" + "="*55)
    print("  Stage 3: Behavioral Probe — Forced Trajectory")
    print("="*55)

    q_before = agent.Q[0].copy()
    print(f"\n  Q(S0, A1) BEFORE: {q_before[0]:.6f}")
    print(f"  Q(S0, A2) BEFORE: {q_before[1]:.6f}")

    state = 0
    action0 = 0
    next_state = 2
    reward = 1.0

    print(f"\n  Trajectory: S0 -[A1(forced)]-> S2 [Rare] -> R=1")

    agent.update(next_state, 0, reward, None)
    agent.update(state, action0, reward, next_state)

    q_after = agent.Q[0].copy()
    print(f"\n  Bellman update for Q(S0, A1):")
    print(f"    Q(S0,A1) += α × [R + γ·V(S2) − Q(S0,A1)]")
    print(f"    Q(S0,A1) += {agent.alpha} × [1.0 + {agent.gamma}×{agent.V[next_state-1]:.6f} − {q_before[0]:.6f}]")
    print(f"\n  Q(S0, A1) AFTER: {q_after[0]:.6f}  (Δ = {q_after[0]-q_before[0]:+.6f})")
    print(f"  Q(S0, A2) AFTER: {q_after[1]:.6f}  (unchanged)")
    print("="*55)


if __name__ == "__main__":
    np.random.seed(42)
    env = TwoStepMDP()
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1)

    print("\n>>> STAGE 2: Temporal Drift Quantification")
    train(env, agent, n_episodes=2000, print_every=400)
    agent.print_q_table("Q-table after 2000 episodes")

    print("\n>>> STAGE 3: Behavioral Probe")
    behavioral_probe(agent)
    agent.print_q_table("Q-table after forced trajectory")