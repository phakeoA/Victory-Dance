"""Self-play: the era loop (generation, league, gate, gate_sim, hof, resume, archive, status, resources, diagnostics,
replay_html), collection (game_runner, mp_collect, mp_eval, spawn_client, spawn_plugin + showdown_plugins/) and the
exploiter meter. The PPO core (ppo, trainer, actor_critic, gae, pbrs, reward, schema, collector, store, value_space,
policy_eval) lives in ``v_dance.rl`` since refactor Phase 2 (2026-09-10); ``ladder_update`` is ``v_dance.ladder.update``.
The old ``v_dance.selfplay.<module>`` paths are ``sys.modules`` shims until Phase 6."""
