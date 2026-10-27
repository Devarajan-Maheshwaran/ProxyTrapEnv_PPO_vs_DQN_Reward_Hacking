from pathlib import Path

import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from proxy_trap import ProxyTrapEnv


TRAP_REWARDS = (0.1, 0.3, 1.0)
TRAIN_STEPS = 5_000
EVAL_EPISODES = 20
RESULTS_DIR = Path("results")


def evaluate(model, trap_reward):
    proxy_returns = []
    true_returns = []
    successes = []

    for episode in range(EVAL_EPISODES):
        env = ProxyTrapEnv(reward_mode="proxy", trap_reward=trap_reward)
        observation, _ = env.reset(seed=30_000 + episode)
        proxy_total = 0.0
        true_total = 0.0

        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(int(action))
            proxy_total += info["proxy_reward"]
            true_total += info["true_reward"]
            if terminated or truncated:
                break

        proxy_returns.append(proxy_total)
        true_returns.append(true_total)
        successes.append(float(true_total > 0.0))
        env.close()

    return {
        "proxy": sum(proxy_returns) / EVAL_EPISODES,
        "true": sum(true_returns) / EVAL_EPISODES,
        "success": sum(successes) / EVAL_EPISODES,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for trap_reward in TRAP_REWARDS:
        env = ProxyTrapEnv(reward_mode="proxy", trap_reward=trap_reward)
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=64,
            batch_size=64,
            policy_kwargs={"net_arch": [32, 32]},
            seed=0,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=TRAIN_STEPS)
        metrics = evaluate(model, trap_reward)
        rows.append({"trap_reward": trap_reward, **metrics})
        print(
            f'trap_reward={trap_reward:.1f} proxy={metrics["proxy"]:.2f} '
            f'true={metrics["true"]:.2f} success={metrics["success"]:.2f}'
        )
        env.close()

    x_values = [row["trap_reward"] for row in rows]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(x_values, [row["proxy"] for row in rows], marker="o", label="Proxy")
    axis.plot(x_values, [row["true"] for row in rows], marker="o", label="True")
    axis.set_xlabel("Trap reward per step")
    axis.set_ylabel("Mean evaluation return")
    axis.set_title("Optimization pressure sweep")
    axis.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pressure_sweep.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()