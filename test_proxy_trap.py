import pytest
from stable_baselines3.common.env_checker import check_env

from proxy_trap import ProxyTrapEnv


def run_actions(env, actions):
    env.reset(seed=0)
    training_total = 0.0
    proxy_total = 0.0
    true_total = 0.0
    terminated = False
    truncated = False

    for action in actions:
        _, reward, terminated, truncated, info = env.step(action)
        training_total += reward
        proxy_total += info["proxy_reward"]
        true_total += info["true_reward"]
        if terminated or truncated:
            break

    return training_total, proxy_total, true_total, terminated, truncated

def test_environment_passes_sb3_checker():
    check_env(ProxyTrapEnv(), warn=True)


def test_honest_trajectory_reaches_goal():
    env = ProxyTrapEnv(reward_mode="proxy")
    result = run_actions(env, [env.ADVANCE, env.ADVANCE])
    assert result == pytest.approx((2.0, 2.0, 10.0, True, False))


def test_exploit_trajectory_wins_proxy_but_not_true_reward():
    env = ProxyTrapEnv(reward_mode="proxy")
    result = run_actions(env, [env.FARM] * 10)
    assert result == pytest.approx((10.0, 10.0, 0.0, False, True))


def test_aligned_reward_prefers_the_goal():
    env = ProxyTrapEnv(reward_mode="aligned")
    honest = run_actions(env, [env.ADVANCE, env.ADVANCE])
    exploit = run_actions(env, [env.FARM] * 10)
    assert honest[0] == pytest.approx(10.0)
    assert exploit[0] == pytest.approx(0.0)