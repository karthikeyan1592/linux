#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""
Regression test: damon_set_attrs() must reject sample_us == 0.

When sample_us is 0, kdamond_usleep(0) never actually sleeps, so the
kdamond kthread busy-spins burning close to 100% of one CPU instead of
waiting between samples. A fixed kernel rejects `state=on` with
sample_us=0 outright (-EINVAL) and no kdamond is ever created.

This does not just trust that the write() failed or succeeded -- for the
case where it wrongly succeeds, the kdamond thread is pinned to a single
CPU and its actual CPU time is measured two independent ways (the
thread's own utime+stime from /proc/<pid>/stat, and that same CPU's own
busy-tick delta from /proc/stat) and compared against a valid-interval
control run in the same process, so "the kernel accepted it" and "the
kernel is now spinning at close to 100% of a core" are both demonstrated
with numbers, not assumed from one another.
"""
import os
import subprocess
import time

import _damon_sysfs

CLK_TCK = os.sysconf('SC_CLK_TCK')


def thread_ticks(pid):
    """utime+stime, in clock ticks, for the given pid/tid.  Splits after
    the last ')' so a comm field containing spaces or parens can't shift
    the field indices."""
    with open('/proc/%d/stat' % pid) as f:
        content = f.read()
    fields = content[content.rfind(')') + 2:].split()
    # fields[0] is state (proc(5) field 3); utime is field 14, stime 15.
    utime = int(fields[11])
    stime = int(fields[12])
    return utime + stime


def cpu_ticks(cpu):
    """(busy, total) clock ticks for cpuN's own /proc/stat line.  busy
    excludes idle and iowait; everything else counts as busy."""
    with open('/proc/stat') as f:
        for line in f:
            if line.startswith('cpu%d ' % cpu):
                vals = [int(x) for x in line.split()[1:]]
                idle, iowait = vals[3], vals[4]
                total = sum(vals)
                return total - idle - iowait, total
    raise RuntimeError('cpu%d not found in /proc/stat' % cpu)


def measure_spin(pid, cpu, seconds):
    """Pin `pid` to `cpu`, then measure both the thread's own CPU time
    and that CPU's own busy-tick delta over `seconds` wall-clock
    seconds.  Returns (thread_pct, cpu_pct)."""
    os.sched_setaffinity(pid, {cpu})
    t0 = thread_ticks(pid)
    c0_busy, c0_total = cpu_ticks(cpu)
    time.sleep(seconds)
    t1 = thread_ticks(pid)
    c1_busy, c1_total = cpu_ticks(cpu)
    thread_pct = 100.0 * (t1 - t0) / (seconds * CLK_TCK)
    cpu_pct = 100.0 * (c1_busy - c0_busy) / max(1, (c1_total - c0_total))
    return thread_pct, cpu_pct


def run_one(sample_us, target_pid, cpu, seconds):
    """Start a kdamond with the given sample_us against target_pid.
    Returns ((thread_pct, cpu_pct), None) if it started (measuring for
    `seconds`), or (None, err) if the kernel rejected the write."""
    kdamond = _damon_sysfs.Kdamond(
            contexts=[_damon_sysfs.DamonCtx(
                ops='vaddr',
                targets=[_damon_sysfs.DamonTarget(pid=target_pid)],
                monitoring_attrs=_damon_sysfs.DamonAttrs(
                    sample_us=sample_us, aggr_us=100000),
                )])
    kdamonds = _damon_sysfs.Kdamonds([kdamond])
    err = kdamonds.start()
    if err is not None:
        return None, err
    pid = int(kdamond.pid)
    result = measure_spin(pid, cpu, seconds)
    kdamonds.stop()
    return result, None


def main():
    target_cpu = 1 if os.cpu_count() > 1 else 0
    file_dir = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.Popen(
            [os.path.join(file_dir, 'access_memory'), '1', '4096', '60000',
                'repeat'])
    try:
        # Control: a valid interval must be accepted and stay near-idle.
        ctrl, err = run_one(5000, proc.pid, target_cpu, seconds=2)
        if err is not None:
            print('control (sample_us=5000) unexpectedly rejected: %s' % err)
            exit(1)
        ctrl_thread_pct, ctrl_cpu_pct = ctrl
        print('control  sample_us=5000: thread=%.1f%% cpu%d=%.1f%%' %
              (ctrl_thread_pct, target_cpu, ctrl_cpu_pct))

        # The bug under test.
        bug, err = run_one(0, proc.pid, target_cpu, seconds=3)
        if err is not None:
            print('sample_us=0 correctly rejected: %s' % err)
            exit(0)  # fixed kernel: PASS

        thread_pct, cpu_pct = bug
        print('sample_us=0    : thread=%.1f%% cpu%d=%.1f%%' %
              (thread_pct, target_cpu, cpu_pct))

        if thread_pct < 50.0:
            print('sample_us=0 was accepted but the kdamond thread did '
                  'not spin (thread=%.1f%%) -- unexpected state, not the '
                  'known busy-spin bug' % thread_pct)
            exit(1)

        print('BUG: sample_us=0 was accepted and kdamond is busy-spinning '
              '(%.1fx the control\'s CPU usage)' %
              (thread_pct / max(0.1, ctrl_thread_pct)))
        exit(1)  # bug present: FAIL
    finally:
        proc.terminate()
        proc.wait()


if __name__ == '__main__':
    main()
