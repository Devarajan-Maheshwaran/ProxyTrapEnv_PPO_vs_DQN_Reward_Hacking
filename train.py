from pathlib import Path

from stable_baselines3 import DQN, PPO

from proxy_trap import ProxyTrapEnv


SEEDS = (0, 1, 2)
ALGORITHMS = ("PPO", "DQN")
REWARD_MODES = ("aligned", "proxy")
TRAIN_STEPS = 5_000
MODEL_DIR = Path("models")
def make_model(algorithm, env, seed):
    shared = {
        "policy": "MlpPolicy",
        "env": env,
        "policy_kwargs": {"net_arch": [32, 32]},
        "seed": seed,
        "device": "cpu",
        "verbose": 0,
    }

    if algorithm == "PPO":
        return PPO(
            **shared,
            n_steps=64,
            batch_size=64,
        )

    return DQN(
        **shared,
        learning_rate=0.001,
        buffer_size=5_000,
        learning_starts=100,
        batch_size=32,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=250,
        exploration_fraction=0.5,
        exploration_final_eps=0.05,
    )
def proxy_only_score(model, episodes=10):
    scores = []
    for episode in range(episodes):
        env = ProxyTrapEnv(reward_mode="proxy")
        observation, _ = env.reset(seed=10_000 + episode)
        proxy_total = 0.0

        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(int(action))
            proxy_total += info["proxy_reward"]
            if terminated or truncated:
                break

        scores.append(proxy_total)
        env.close()

    return sum(scores) / len(scores)
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    completed = 0

    for algorithm in ALGORITHMS:
        for reward_mode in REWARD_MODES:
            for seed in SEEDS:
                env = ProxyTrapEnv(reward_mode=reward_mode)
                model = make_model(algorithm, env, seed)
                model.learn(total_timesteps=TRAIN_STEPS)

                model_name = f"{algorithm.lower()}_{reward_mode}_seed{seed}"
                model.save(MODEL_DIR / model_name)
                score = proxy_only_score(model)
                completed += 1

                print(
                    f"{completed:02}/12 saved {model_name}.zip "
                    f"proxy-only score={score:.2f}"
                )
                env.close()

    print("Proxy-only ranking complete. True performance is still unknown.")


if __name__ == "__main__":
    main()