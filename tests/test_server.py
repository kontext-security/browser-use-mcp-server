from typing import Any

import pytest

from server import server


@pytest.mark.asyncio
async def test_browser_session_uses_current_configuration(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    class FakeBrowserSession:
        def __init__(self, **kwargs: Any):
            captured.update(kwargs)

    monkeypatch.setattr(server, "BrowserSession", FakeBrowserSession)

    session = await server.create_browser_context_for_task(
        chrome_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        window_width=1440,
        window_height=900,
        locale="de-DE",
    )

    assert isinstance(session, FakeBrowserSession)
    assert captured["executable_path"].endswith("Google Chrome")
    assert captured["window_size"] == {"width": 1440, "height": 900}
    assert captured["enable_default_extensions"] is False
    assert captured["captcha_solver"] is False
    assert "--lang=de-DE" in captured["args"]
    assert not any(
        arg.startswith("--remote-debugging-port=") for arg in captured["args"]
    )
    assert captured["minimum_wait_page_load_time"] == 0.2
    assert captured["wait_for_network_idle_page_load_time"] == 0.6


@pytest.mark.asyncio
async def test_task_passes_session_to_agent_and_stops_it(
    monkeypatch: pytest.MonkeyPatch,
):
    task_id = "modern-browser-session"

    class FakeSession:
        stopped = False

        async def stop(self):
            self.stopped = True

    fake_session = FakeSession()
    agent_kwargs: dict[str, Any] = {}

    async def fake_create_session(**_kwargs: Any):
        return fake_session

    class FakeHistory:
        def final_result(self):
            return "done"

        def is_successful(self):
            return True

        def has_errors(self):
            return False

        def errors(self):
            return []

        def urls(self):
            return ["https://example.com"]

        def action_names(self):
            return ["navigate"]

        def extracted_content(self):
            return []

        def number_of_steps(self):
            return 1

    class FakeAgent:
        def __init__(self, **kwargs: Any):
            agent_kwargs.update(kwargs)

        async def run(self, max_steps: int):
            assert max_steps == server.CONFIG["MAX_AGENT_STEPS"]
            return FakeHistory()

    monkeypatch.setattr(server, "create_browser_context_for_task", fake_create_session)
    monkeypatch.setattr(server, "Agent", FakeAgent)
    server.task_store[task_id] = {"status": "pending"}

    try:
        await server.run_browser_task_async(
            task_id=task_id,
            url="https://example.com",
            action="read the title",
            llm=object(),
        )

        assert agent_kwargs["browser_session"] is fake_session
        assert "browser_context" not in agent_kwargs
        assert fake_session.stopped is True
        assert server.task_store[task_id]["status"] == "completed"
        assert server.task_store[task_id]["result"]["final_result"] == "done"
    finally:
        server.task_store.pop(task_id, None)
