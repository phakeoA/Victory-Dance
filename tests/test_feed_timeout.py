"""2026-09-29 live bug: one host.feed_async that never returns froze the whole online consumer. The consumer now
awaits the feed with a deadline; this checks the mechanism it relies on (a hung POKE_LOOP coroutine is abandoned,
cancelled, and reported) without a browser."""
import asyncio

from poke_env.concurrency import POKE_LOOP

import v_dance.online.play_vs_human_browser as pvhb


def test_hung_feed_times_out_is_cancelled_and_reported(capsys):
    started = asyncio.run_coroutine_threadsafe(asyncio.sleep(0), POKE_LOOP)
    started.result(timeout=5)
    cancelled = []

    async def hung_feed():
        try:
            await asyncio.Event().wait()          # never completes — like the 16:06 UTC hang
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    async def consumer_step():
        fut = asyncio.run_coroutine_threadsafe(hung_feed(), POKE_LOOP)
        try:
            await asyncio.wait_for(asyncio.wrap_future(fut), 0.3)
            return "returned"
        except asyncio.TimeoutError:
            fut.cancel()
            pvhb._report_stuck_feed("battle-test-1", ">battle-test-1\n|request|{}\n")
            return "timed out"

    pvhb._stuck_feeds = 0
    assert asyncio.run(consumer_step()) == "timed out"
    for _ in range(50):                            # the cancel lands on POKE_LOOP's thread
        if cancelled:
            break
        asyncio.run(asyncio.sleep(0.02))
    assert cancelled == [True]
    out = capsys.readouterr().out
    assert "FEED STUCK" in out and "battle-test-1" in out
    pvhb._stuck_feeds = 0
