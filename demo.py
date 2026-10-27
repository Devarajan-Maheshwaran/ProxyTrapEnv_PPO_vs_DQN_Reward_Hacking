from proxy_trap import ProxyTrapEnv


def run_sequence(label, reward_mode, actions):
    env = ProxyTrapEnv(reward_mode=reward_mode)
    env.reset(seed=0)
    training_total = 0.0
    proxy_total = 0.0
    true_total = 0.0
    trace = [env.render()]

    for action in actions:
        _, reward, terminated, truncated, info = env.step(action)
        training_total += reward
        proxy_total += info["proxy_reward"]
        true_total += info["true_reward"]
        trace.append(env.render())
        if terminated or truncated:
            break

    print(
        f"{reward_mode:7} | {label:7} | training={training_total:4.1f} "
        f"proxy={proxy_total:4.1f} true={true_total:4.1f}"
    )
    print("  " + " -> ".join(trace))

def main():
    for reward_mode in ("proxy", "aligned"):
        run_sequence(
            "honest",
            reward_mode,
            [ProxyTrapEnv.ADVANCE, ProxyTrapEnv.ADVANCE],
        )
        run_sequence(
            "exploit",
            reward_mode,
            [ProxyTrapEnv.FARM] * 10,
        )


if __name__ == "__main__":
    main()