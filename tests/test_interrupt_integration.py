# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from colab_cli.cli import app
from colab_cli.runtime import ColabRuntime
from colab_cli.state import SessionState

runner = CliRunner()


@pytest.fixture
def mock_store(mock_common_state):
    return mock_common_state.store


def test_integration_loop_interrupted_by_cli_interrupt(
    mock_store, mock_common_state, mocker
):
    """
    Simulates a concurrent execution where a long-running/looping cell is executing
    in one thread, and `colab interrupt -s <session>` is called from another thread.
    Verifies that the interrupt signals the running execution, terminates the loop,
    and returns cleanly.
    """
    real_session = SessionState(
        name="sess-loop",
        url="http://fake-colab/api",
        endpoint="endpoint-xyz",
        token="token-xyz",
        kernel_id="kernel-1",
        session_id="session-1",
    )

    mock_store.get.return_value = real_session
    mock_common_state.resolve_session.return_value = "sess-loop"

    loop_started = threading.Event()
    interrupted_event = threading.Event()

    # conftest.py mocks ColabRuntime in commands.execution and commands.session
    mock_exec_runtime_cls = mocker.patch("colab_cli.commands.execution.ColabRuntime")
    mock_session_runtime_cls = mocker.patch("colab_cli.commands.session.ColabRuntime")

    exec_runtime_instance = MagicMock()
    session_runtime_instance = MagicMock()

    mock_exec_runtime_cls.return_value = exec_runtime_instance
    mock_session_runtime_cls.return_value = session_runtime_instance

    def mock_execute_code(code, output_hook=None, timeout=None):
        if "while True" in code:
            loop_started.set()
            # Wait until interrupt arrives
            signaled = interrupted_event.wait(timeout=5.0)
            if signaled and output_hook:
                output_hook(
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": "KeyboardInterrupt: Cell interrupted by user\n",
                    }
                )
            return [
                {
                    "output_type": "error",
                    "ename": "KeyboardInterrupt",
                    "evalue": "Cell interrupted by user",
                }
            ]
        return []

    def mock_interrupt(timeout=None):
        interrupted_event.set()

    exec_runtime_instance.execute_code.side_effect = mock_execute_code
    session_runtime_instance.interrupt.side_effect = mock_interrupt

    exec_results = []

    def run_exec():
        res = runner.invoke(app, ["exec", "-s", "sess-loop"], input="while True:\n    pass\n")
        exec_results.append(res)

    exec_thread = threading.Thread(target=run_exec)
    exec_thread.start()

    # Ensure exec thread is actively executing inside the loop
    assert loop_started.wait(timeout=3.0), "Exec loop failed to start in time"

    # Issue colab interrupt from another caller
    int_res = runner.invoke(app, ["interrupt", "-s", "sess-loop"])
    assert int_res.exit_code == 0
    assert "Interrupted kernel for session 'sess-loop'." in int_res.output

    # Ensure the exec thread unblocks and terminates cleanly
    exec_thread.join(timeout=3.0)
    assert not exec_thread.is_alive(), "Exec thread failed to exit after interrupt"

    assert len(exec_results) == 1
    exec_res = exec_results[0]
    assert exec_res.exit_code == 0
    assert "KeyboardInterrupt" in exec_res.output
    session_runtime_instance.interrupt.assert_called_once()
