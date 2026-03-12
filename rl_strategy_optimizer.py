"""
=============================================================================
  RL STRATEGY OPTIMIZER — Corporate Bankruptcy Prediction System
=============================================================================
  Reinforcement Learning module that trains a PPO agent to learn financial
  adjustment strategies that minimize bankruptcy probability.

  Components:
    1. FinancialRiskEnv  — Gymnasium-compatible custom environment
    2. train_rl_agent    — PPO training via Stable-Baselines3
    3. evaluate_strategy — Demonstrates learned strategy step-by-step
    4. generate_rl_outputs — Generates all RL visualizations & reports

  The RL agent interacts with the trained XGBoost model as a "financial
  risk simulator" — it is NOT making real financial decisions.
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from config import (
    BASE_DIR, OUTPUT_DIR, DIAGNOSTICS_DIR, FIGURE_DPI,
    RANDOM_SEED, DATA_PATH, TARGET_COLUMN
)
from risk_simulator import predict_bankruptcy_risk


# ──────────────────────────────────────────────────────────────────────────────
#  STYLE
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.6,
    'font.family': 'sans-serif',
    'font.size': 11,
})

RL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "rl_outputs")


# ──────────────────────────────────────────────────────────────────────────────
#  LOAD FEATURE METADATA
# ──────────────────────────────────────────────────────────────────────────────

def load_feature_metadata():
    """Load feature metadata from JSON."""
    path = os.path.join(BASE_DIR, "feature_metadata.json")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['features']


def load_sample_states(n_samples=100):
    """Load random financial states from the dataset."""
    df = pd.read_csv(DATA_PATH)
    features = df.drop(columns=[TARGET_COLUMN])
    indices = np.random.choice(len(features), size=min(n_samples, len(features)),
                                replace=False)
    return features.iloc[indices].values


# ══════════════════════════════════════════════════════════════════════════════
#  1. CUSTOM GYMNASIUM ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class FinancialRiskEnv(gym.Env):
    """
    Financial Risk Reduction Environment.

    The agent adjusts a company's 48 financial indicators to reduce
    bankruptcy probability over multiple steps (simulated quarters).

    State:  48-dimensional continuous vector (financial features)
    Action: 48-dimensional continuous vector (adjustments to features)
    Reward: Reduction in bankruptcy probability
    """

    metadata = {'render_modes': []}

    def __init__(self, sample_states=None, max_steps=10,
                 feature_metadata=None):
        super().__init__()

        self.max_steps = max_steps
        self.current_step = 0
        self.n_features = 48

        # Feature metadata for realistic adjustments
        if feature_metadata is None:
            feature_metadata = load_feature_metadata()
        self.feature_metadata = feature_metadata

        # Pre-compute adjustment ranges
        self.adj_low = np.array([f['adjustment_range'][0] for f in feature_metadata])
        self.adj_high = np.array([f['adjustment_range'][1] for f in feature_metadata])

        # Scale down adjustments per step (10% of full range per step)
        self.step_scale = 0.10
        self.step_low = self.adj_low * self.step_scale
        self.step_high = self.adj_high * self.step_scale

        # Sample states from dataset
        if sample_states is not None:
            self.sample_states = sample_states
        else:
            self.sample_states = load_sample_states(200)

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_features,), dtype=np.float32
        )

        # Action: normalized [-1, 1] per feature, then scaled to adjustment range
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_features,), dtype=np.float32
        )

        # State tracking
        self.state = None
        self.prev_probability = None
        self.initial_probability = None
        self.history = []
        self.action_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick a random company from dataset
        idx = np.random.randint(0, len(self.sample_states))
        self.state = self.sample_states[idx].copy().astype(np.float32)

        self.current_step = 0
        self.initial_probability = predict_bankruptcy_risk(self.state)
        self.prev_probability = self.initial_probability
        self.history = [{
            'step': 0,
            'probability': self.initial_probability,
            'action': 'initial',
            'state': self.state.copy()
        }]
        self.action_history = []

        return self.state.copy(), {}

    def step(self, action):
        self.current_step += 1

        # Scale action from [-1, 1] to realistic adjustment ranges
        adjustments = np.where(
            action >= 0,
            action * self.step_high,
            -action * self.step_low
        ).astype(np.float32)

        # ── ENFORCE DIVERSITY ──
        # Find the primary action the agent is trying to take
        top_action_idx = np.argmax(np.abs(adjustments))
        
        # If repeated more than twice, zero it out and find the next best
        if len(self.action_history) >= 2 and self.action_history[-1] == top_action_idx and self.action_history[-2] == top_action_idx:
            adjustments[top_action_idx] = 0.0
            action[top_action_idx] = 0.0
            top_action_idx = np.argmax(np.abs(adjustments))
            
        self.action_history.append(top_action_idx)

        # Apply adjustments to state
        self.state = self.state + adjustments

        # Get new bankruptcy probability
        new_probability = predict_bankruptcy_risk(self.state)

        # ── REWARD FUNCTION ──
        # Primary: probability reduction
        prob_reduction = self.prev_probability - new_probability

        # Bonus for reaching low-risk zones
        zone_bonus = 0.0
        if new_probability < 0.20:
            zone_bonus = 0.3
        elif new_probability < 0.40:
            zone_bonus = 0.1

        # Penalty for large reckless adjustments
        action_penalty = -0.01 * np.sum(np.abs(action))

        reward = float(prob_reduction * 10.0 + zone_bonus + action_penalty)

        # Track history
        action_name = self.feature_metadata[top_action_idx]['name']
        self.history.append({
            'step': self.current_step,
            'probability': new_probability,
            'action': action_name,
            'adjustment': float(adjustments[top_action_idx]),
            'state': self.state.copy()
        })

        self.prev_probability = new_probability

        # Termination
        terminated = (self.current_step >= self.max_steps)
        truncated = False

        return self.state.copy(), reward, terminated, truncated, {}


# ══════════════════════════════════════════════════════════════════════════════
#  2. REWARD TRACKING CALLBACK
# ══════════════════════════════════════════════════════════════════════════════

class RewardTracker(BaseCallback):
    """Track episode rewards during training."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = 0.0

    def _on_step(self):
        # Accumulate rewards
        if len(self.locals.get('rewards', [])) > 0:
            self.current_rewards += self.locals['rewards'][0]

        # Check for episode end
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            self.episode_rewards.append(self.current_rewards)
            self.current_rewards = 0.0
        return True


# ══════════════════════════════════════════════════════════════════════════════
#  3. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_rl_agent(total_timesteps=50000):
    """
    Train PPO agent to learn financial risk reduction strategies.

    Returns
    -------
    model : PPO
        Trained RL model.
    tracker : RewardTracker
        Contains episode rewards for plotting.
    env : FinancialRiskEnv
        The environment (for evaluation).
    """
    print("=" * 70)
    print("  🎮  REINFORCEMENT LEARNING — PPO Strategy Optimizer")
    print("=" * 70)
    print()

    # Load states and metadata
    print("  ⏳  Loading sample financial states from dataset...")
    sample_states = load_sample_states(500)
    feature_metadata = load_feature_metadata()

    print(f"  📊  Sample states loaded: {len(sample_states)}")
    print(f"  📊  Features per state:   {sample_states.shape[1]}")
    print()

    # Create environment
    env = FinancialRiskEnv(
        sample_states=sample_states,
        max_steps=10,
        feature_metadata=feature_metadata
    )

    # Initialize PPO
    print("  🤖  Initializing PPO Agent...")
    print(f"       Policy:       MlpPolicy")
    print(f"       Timesteps:    {total_timesteps:,}")
    print(f"       Learning Rate: 3e-4")
    print(f"       Batch Size:   64")
    print(f"       Episodes:     10 steps each (simulated quarters)")
    print()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=RANDOM_SEED,
    )

    # Training with reward tracking
    tracker = RewardTracker()
    print(f"  ⏳  Training PPO for {total_timesteps:,} timesteps...")
    print(f"       (This may take 1–3 minutes)")
    print()

    model.learn(total_timesteps=total_timesteps, callback=tracker)

    print(f"  ✅  Training complete!")
    if len(tracker.episode_rewards) > 0:
        print(f"  📊  Total episodes:    {len(tracker.episode_rewards)}")
        print(f"  📊  Mean reward (last 100): "
              f"{np.mean(tracker.episode_rewards[-100:]):.4f}")
        print(f"  📊  Max episode reward: {max(tracker.episode_rewards):.4f}")
    print()

    return model, tracker, env


# ══════════════════════════════════════════════════════════════════════════════
#  4. STRATEGY EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_strategy(model, env, n_episodes=20):
    """
    Run the trained agent and collect strategies.

    Returns
    -------
    strategies : list of dicts
        Each dict contains episode history.
    """
    print("  📋  Evaluating learned strategies...")

    strategies = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        strategy = {
            'episode': ep,
            'initial_prob': env.history[0]['probability'],
            'final_prob': env.history[-1]['probability'],
            'reduction_pct': (1 - env.history[-1]['probability'] /
                             max(env.history[0]['probability'], 1e-10)) * 100,
            'history': env.history.copy()
        }
        strategies.append(strategy)

    # Report
    reductions = [s['reduction_pct'] for s in strategies]
    print(f"  📊  Mean risk reduction: {np.mean(reductions):.1f}%")
    print(f"  📊  Best reduction:      {max(reductions):.1f}%")
    print(f"  📊  Worst reduction:     {min(reductions):.1f}%")
    print()

    return strategies


# ══════════════════════════════════════════════════════════════════════════════
#  5. OUTPUT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_rl_outputs(tracker, strategies, feature_metadata):
    """Generate all RL visualizations and reports."""

    os.makedirs(RL_OUTPUT_DIR, exist_ok=True)

    # ── 5a. Training Reward Curve ──
    print("  📊  Generating RL training rewards plot...")
    if len(tracker.episode_rewards) > 0:
        fig, ax = plt.subplots(figsize=(14, 7))

        episodes = range(len(tracker.episode_rewards))
        ax.plot(episodes, tracker.episode_rewards, alpha=0.3,
                color='#58a6ff', linewidth=0.8, label='Episode Reward')

        # Smoothed curve (moving average)
        window = min(50, len(tracker.episode_rewards) // 5 + 1)
        if window > 1:
            smoothed = pd.Series(tracker.episode_rewards).rolling(
                window=window, min_periods=1).mean()
            ax.plot(episodes, smoothed, color='#f78166',
                    linewidth=2.5, label=f'Moving Avg ({window} eps)')

        ax.set_title('RL Training — Episode Rewards',
                     fontweight='bold', fontsize=15, pad=15, color='#c9d1d9')
        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Total Reward', fontweight='bold')
        ax.legend(fontsize=11, framealpha=0.8)
        ax.grid(True, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

        plt.tight_layout()
        path = os.path.join(RL_OUTPUT_DIR, 'rl_training_rewards.png')
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight',
                    facecolor='#0d1117')
        plt.close()
        print(f"  💾  Saved: {path}")

    # ── 5b. Policy Actions Analysis ──
    print("  📊  Analyzing learned policy actions...")
    action_counts = {}
    for strategy in strategies:
        for step_data in strategy['history'][1:]:  # skip initial
            action = step_data.get('action', 'unknown')
            if action not in action_counts:
                action_counts[action] = {
                    'count': 0,
                    'total_adjustment': 0.0,
                }
            action_counts[action]['count'] += 1
            action_counts[action]['total_adjustment'] += abs(
                step_data.get('adjustment', 0))

    actions_df = pd.DataFrame([
        {
            'Feature_Adjusted': name,
            'Times_Selected': data['count'],
            'Mean_Adjustment': data['total_adjustment'] / max(data['count'], 1),
        }
        for name, data in action_counts.items()
    ]).sort_values('Times_Selected', ascending=False)

    path = os.path.join(RL_OUTPUT_DIR, 'rl_policy_actions.csv')
    actions_df.to_csv(path, index=False)
    print(f"  💾  Saved: {path}")

    # ── 5c. Strategy Example Report ──
    print("  📊  Generating strategy example report...")

    # Find the best strategy (highest risk reduction)
    best = max(strategies, key=lambda s: s['reduction_pct'])

    path = os.path.join(RL_OUTPUT_DIR, 'rl_strategy_example.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 65 + "\n")
        f.write("  RL STRATEGY EXAMPLE — Best Risk Reduction Path\n")
        f.write("  Algorithm: PPO (Proximal Policy Optimization)\n")
        f.write("  Environment: FinancialRiskEnv (10-step episodes)\n")
        f.write("=" * 65 + "\n\n")

        f.write(f"  Initial Probability: {best['initial_prob']:.4f} "
                f"({best['initial_prob']*100:.2f}%)\n\n")

        for step_data in best['history'][1:]:
            step = step_data['step']
            prob = step_data['probability']
            action = step_data['action']
            adj = step_data.get('adjustment', 0)
            direction = "increase" if adj > 0 else "reduce"
            f.write(f"  Step {step:>2}: {direction} {action[:45]}\n")
            f.write(f"           → Probability: {prob:.4f} "
                    f"({prob*100:.2f}%)\n\n")

        final_prob = best['history'][-1]['probability']
        f.write(f"  {'─' * 55}\n")
        f.write(f"  Final Probability: {final_prob:.4f} "
                f"({final_prob*100:.2f}%)\n")
        f.write(f"  Risk Reduction:    {best['reduction_pct']:.1f}%\n\n")

        # Risk category mapping
        initial_score = best['initial_prob'] * 100
        final_score = final_prob * 100

        def get_category(score):
            if score < 20: return "🟢 Safe"
            elif score < 40: return "🔵 Low Risk"
            elif score < 60: return "🟡 Moderate Risk"
            elif score < 80: return "🟠 High Risk"
            else: return "🔴 Critical Risk"

        f.write(f"  Initial Risk: {get_category(initial_score)} "
                f"(Score: {initial_score:.1f})\n")
        f.write(f"  Final Risk:   {get_category(final_score)} "
                f"(Score: {final_score:.1f})\n")

    print(f"  💾  Saved: {path}")

    # ── 5d. Risk Reduction Simulation Plot ──
    print("  📊  Generating risk reduction simulation plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Left: Best strategy trajectory
    steps = [h['step'] for h in best['history']]
    probs = [h['probability'] * 100 for h in best['history']]

    # Color zones
    ax1.axhspan(0, 20, alpha=0.08, color='#3fb950', label='Safe Zone')
    ax1.axhspan(20, 40, alpha=0.08, color='#58a6ff')
    ax1.axhspan(40, 60, alpha=0.08, color='#d29922')
    ax1.axhspan(60, 80, alpha=0.08, color='#f78166')
    ax1.axhspan(80, 100, alpha=0.08, color='#f85149')

    ax1.plot(steps, probs, 'o-', color='#58a6ff', linewidth=2.5,
             markersize=8, markeredgecolor='#c9d1d9', markeredgewidth=1.5,
             zorder=5)

    # Annotate start and end
    ax1.annotate(f'{probs[0]:.1f}%', (steps[0], probs[0]),
                textcoords="offset points", xytext=(10, 10),
                fontweight='bold', fontsize=11, color='#f85149')
    ax1.annotate(f'{probs[-1]:.1f}%', (steps[-1], probs[-1]),
                textcoords="offset points", xytext=(10, -15),
                fontweight='bold', fontsize=11, color='#3fb950')

    ax1.set_title('Best Strategy — Risk Reduction Path',
                 fontweight='bold', fontsize=14, pad=15, color='#c9d1d9')
    ax1.set_xlabel('Step (Simulated Quarter)', fontweight='bold')
    ax1.set_ylabel('Bankruptcy Risk Score (%)', fontweight='bold')
    ax1.set_ylim([-5, 105])
    ax1.grid(True, alpha=0.3)
    for spine in ax1.spines.values():
        spine.set_edgecolor('#30363d')

    # Right: Multiple strategy trajectories
    for i, strategy in enumerate(strategies[:10]):
        s_steps = [h['step'] for h in strategy['history']]
        s_probs = [h['probability'] * 100 for h in strategy['history']]
        alpha = 0.6 if i > 0 else 1.0
        ax2.plot(s_steps, s_probs, 'o-', alpha=alpha, linewidth=1.5,
                markersize=4)

    ax2.axhspan(0, 20, alpha=0.08, color='#3fb950')
    ax2.set_title('Multiple Companies — RL Risk Reduction',
                 fontweight='bold', fontsize=14, pad=15, color='#c9d1d9')
    ax2.set_xlabel('Step (Simulated Quarter)', fontweight='bold')
    ax2.set_ylabel('Bankruptcy Risk Score (%)', fontweight='bold')
    ax2.set_ylim([-5, 105])
    ax2.grid(True, alpha=0.3)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#30363d')

    plt.tight_layout()
    path = os.path.join(RL_OUTPUT_DIR, 'rl_risk_reduction_simulation.png')
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close()
    print(f"  💾  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_rl_simulation(total_timesteps=50000):
    """
    Complete RL pipeline: train, evaluate, generate outputs.

    Returns
    -------
    strategies : list
        Evaluated strategies.
    """
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║   🎮  REINFORCEMENT LEARNING STRATEGY SIMULATOR                  ║")
    print("║   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                 ║")
    print("║   Algorithm: PPO (Proximal Policy Optimization)                   ║")
    print("║   Framework: Stable-Baselines3 + Gymnasium                        ║")
    print("║   Objective: Learn strategies to reduce bankruptcy risk            ║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # 1. Train
    model, tracker, env = train_rl_agent(total_timesteps)

    # 2. Evaluate
    strategies = evaluate_strategy(model, env, n_episodes=20)

    # 3. Generate outputs
    feature_metadata = load_feature_metadata()
    generate_rl_outputs(tracker, strategies, feature_metadata)

    # 4. Summary
    best = max(strategies, key=lambda s: s['reduction_pct'])
    mean_reduction = np.mean([s['reduction_pct'] for s in strategies])

    print()
    print("=" * 70)
    print("  ✅  RL STRATEGY SIMULATION COMPLETE")
    print("=" * 70)
    print(f"  📊  Best risk reduction:   {best['reduction_pct']:.1f}%")
    print(f"  📊  Mean risk reduction:   {mean_reduction:.1f}%")
    print(f"  📊  Best: {best['initial_prob']*100:.1f}% → "
          f"{best['history'][-1]['probability']*100:.1f}%")
    print(f"  📁  Outputs saved to: {RL_OUTPUT_DIR}")
    print()

    return strategies
