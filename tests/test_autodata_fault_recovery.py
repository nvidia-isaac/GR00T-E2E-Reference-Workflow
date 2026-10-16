# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Recovery cancels interrupted work without exporting misaligned recordings."""

import asyncio
from types import SimpleNamespace

import pytest

from e2e_robot_workflows.autodata.arena import run_generation_with_recovery
from e2e_robot_workflows.controllers.errors import H2MeasuredJointLimitViolation


@pytest.mark.parametrize("fault_count", [1, 3])
@pytest.mark.parametrize("exhaust_budget", [False, True])
def test_measured_faults_cancel_worker_and_use_fresh_queues(fault_count, exhaust_budget):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stats = dict(
        num_attempts_started=0, num_attempts_completed=0, num_attempts=0, num_success=0, num_failures=0, num_aborted=0
    )
    queues = []
    resets = []
    cancelled = []
    reports = []

    async def worker(reset_queue, action_queue):
        if exhaust_budget and stats["num_attempts_started"] == fault_count:
            return
        stats["num_attempts_started"] += 1
        try:
            await action_queue.put((0, "action"))
            await action_queue.join()
        finally:
            cancelled.append(True)

    def env_loop(env, reset_queue, action_queue, event_loop, **kwargs):
        queues.append((reset_queue, action_queue))
        event_loop.run_until_complete(asyncio.sleep(0))
        if len(queues) <= fault_count:
            raise H2MeasuredJointLimitViolation("measured wrist excursion")
        if exhaust_budget:
            event_loop.run_until_complete(kwargs["data_gen_tasks"])
            return
        action_queue.get_nowait()
        action_queue.task_done()
        event_loop.run_until_complete(kwargs["data_gen_tasks"])
        stats["num_success"] += 1
        stats["num_attempts"] += 1
        stats["num_attempts_completed"] += 1

    try:
        result = {}
        run_generation_with_recovery(
            env=SimpleNamespace(recorder_manager=SimpleNamespace(reset=resets.append)),
            event_loop=loop,
            env_loop=env_loop,
            worker_factory=worker,
            policy=None,
            stats=stats,
            result=result,
            write_result=lambda value: reports.append(dict(value)),
        )
        assert stats["num_aborted"] == fault_count
        successful = 0 if exhaust_budget else 1
        assert stats["num_success"] == successful
        assert stats["num_failures"] == 0
        assert stats["num_attempts_started"] == stats["num_attempts_completed"] == fault_count + successful
        assert stats["num_attempts"] == fault_count + successful
        assert len(cancelled) == fault_count + successful
        assert len(resets) == len(reports) == len(result["aborted_attempts"]) == fault_count
        assert len({id(pair[1]) for pair in queues}) == fault_count + 1
        assert not asyncio.all_tasks(loop)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def test_unrelated_error_is_fatal_and_cancels_waiting_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    cancelled = []

    async def worker(*queues):
        try:
            await asyncio.Future()
        finally:
            cancelled.append(True)

    def env_loop(env, reset_queue, action_queue, event_loop, **kwargs):
        event_loop.run_until_complete(asyncio.sleep(0))
        raise RuntimeError("unrelated failure")

    try:
        with pytest.raises(RuntimeError, match="unrelated failure"):
            run_generation_with_recovery(
                env=None,
                event_loop=loop,
                env_loop=env_loop,
                worker_factory=worker,
                policy=None,
                stats={},
                result={},
                write_result=lambda _: None,
            )
        assert cancelled == [True]
        assert not asyncio.all_tasks(loop)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@pytest.mark.parametrize("primary", [RuntimeError("environment failure"), H2MeasuredJointLimitViolation("wrist limit")])
def test_cleanup_failure_preserves_primary_and_stops_retries(primary):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = {}

    async def worker(*queues):
        try:
            await asyncio.Future()
        finally:
            raise KeyError("worker cleanup")

    def env_loop(env, reset_queue, action_queue, event_loop, **kwargs):
        event_loop.run_until_complete(asyncio.sleep(0))
        raise primary

    try:
        with pytest.raises(type(primary)) as caught:
            run_generation_with_recovery(
                env=None,
                event_loop=loop,
                env_loop=env_loop,
                worker_factory=worker,
                policy=None,
                stats={},
                result=result,
                write_result=lambda _: None,
            )
        assert caught.value is primary
        assert result["cleanup_errors"] == ["generator_task: KeyError: 'worker cleanup'"]
        assert result["cleanup_errors"][0] in primary.__notes__
        assert not asyncio.all_tasks(loop)
    finally:
        loop.close()
        asyncio.set_event_loop(None)
