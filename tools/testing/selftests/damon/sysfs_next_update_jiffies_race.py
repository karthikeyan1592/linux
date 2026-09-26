#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""
Regression test: damon_sysfs_repeat_call_fn()'s next-refresh deadline must
not be shared across kdamonds.

damon_sysfs_repeat_call_fn() reads and writes a next-refresh deadline with
no lock, and runs inside each kdamond's own kthread via kdamond_call(). If
that deadline is a single variable shared by every kdamond (as it was
before the per-kdamond fix), running two kdamonds with refresh_ms set
gives two real threads unsynchronized read/write access to the same word
-- a KCSAN-detectable data race, confirmed via a live KCSAN capture during
this bug's investigation (twice in a 5-second window, first attempt).

kdamond_call() -- which is what actually invokes
damon_sysfs_repeat_call_fn() for a repeating control -- runs once per
AGGREGATION window, not once per sample, so a short sample/aggregation
interval on a trivial 'paddr' region (matching the parameters that
reliably reproduced this originally) gives far more attempts per second
than a realistic 'vaddr' target would in the same test window.
_damon_sysfs.py's DamonTarget class only supports 'vaddr' (pid-based)
targeting, so the small 'paddr' region here is staged directly.

This is a race, not a functional divergence reliable enough to assert on
timing alone, so this test's only real oracle is KCSAN itself: it skips
(ksft_skip) if CONFIG_KCSAN is not detectably enabled, the same way
sysfs_no_op_commit_break.py skips without 'drgn'.
"""
import os
import subprocess
import time

import _damon_sysfs

KCSAN_PARAMS = '/sys/module/kcsan/parameters'
RACE_SIGNATURE = 'data-race in damon_sysfs_repeat_call_fn'
KDAMONDS_DIR = os.path.join(_damon_sysfs.sysfs_root, 'kdamonds')


def kcsan_available():
    return os.path.exists(os.path.join(KCSAN_PARAMS, 'skip_watch'))


def tune_kcsan():
    """Best-effort: lower skip_watch/udelay so a real race is likely to
    be caught within the test's short window, rather than relying on
    default (much less sensitive) parameters."""
    for name, value in (('skip_watch', '32'), ('udelay_task', '40'),
                         ('udelay_interrupt', '10')):
        try:
            with open(os.path.join(KCSAN_PARAMS, name), 'w') as f:
                f.write(value)
        except Exception:
            pass  # not fatal -- worst case, default sensitivity


def clear_dmesg():
    try:
        # '-C' (util-linux) vs '-c' (busybox) differ; try both.
        if subprocess.call(['dmesg', '-C'],
                            stderr=subprocess.DEVNULL) != 0:
            subprocess.call(['dmesg', '-c'], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    except Exception:
        pass  # best-effort; a stale unrelated splat would just be noise


def dmesg_has_race():
    try:
        out = subprocess.check_output(['dmesg'], stderr=subprocess.DEVNULL,
                                       text=True)
    except Exception as e:
        print('failed to read dmesg: %s' % e)
        return None
    return RACE_SIGNATURE in out


def stage_kdamond(idx, refresh_ms):
    """Directly stage a minimal 'paddr' kdamond with a trivial [0, 4096)
    region -- matching the parameters (sample_us=500, aggr_us=5000) that
    reliably reproduced this race originally. Raw sysfs writes, not
    _damon_sysfs.py's DamonTarget: that class only supports 'vaddr'
    (pid-based) targeting, and this race needs a high kdamond_call()
    rate, not a realistic access pattern.
    """
    kd = os.path.join(KDAMONDS_DIR, '%d' % idx)
    ctx = os.path.join(kd, 'contexts', '0')
    steps = [
        (os.path.join(kd, 'contexts', 'nr_contexts'), '1'),
        (os.path.join(ctx, 'operations'), 'paddr'),
        (os.path.join(ctx, 'targets', 'nr_targets'), '1'),
        (os.path.join(ctx, 'targets', '0', 'regions', 'nr_regions'), '1'),
        (os.path.join(ctx, 'targets', '0', 'regions', '0', 'start'), '0'),
        (os.path.join(ctx, 'targets', '0', 'regions', '0', 'end'), '4096'),
        (os.path.join(ctx, 'monitoring_attrs', 'intervals', 'sample_us'),
            '500'),
        (os.path.join(ctx, 'monitoring_attrs', 'intervals', 'aggr_us'),
            '5000'),
        (os.path.join(kd, 'refresh_ms'), '%d' % refresh_ms),
    ]
    # Several of these sysfs store handlers (e.g. nr_regions_store()) use
    # mutex_trylock(&damon_sysfs_lock) -- non-blocking -- and that lock is
    # shared by every kdamond's sysfs operations, including the already-
    # running kdamond.0's own repeat_call_fn() work. A transient EBUSY
    # here is expected lock contention, not a real failure; retry briefly.
    for path, value in steps:
        for attempt in range(10):
            err = _damon_sysfs.write_file(path, value)
            if err is None or 'Errno 16' not in err:
                break
            time.sleep(0.05)
        if err is not None:
            return '%s <- %s' % (err, path)
    # Observed once: 'state'='on' racing kobject creation right after
    # nr_kdamonds is bumped returns EBUSY transiently. Not the bug under
    # test -- a harness timing hiccup -- so retry briefly before giving up.
    for attempt in range(5):
        err = _damon_sysfs.write_file(os.path.join(kd, 'state'), 'on')
        if err is None or 'Errno 16' not in err:
            return err
        time.sleep(0.2)
    return err


def stop_kdamond(idx):
    kd = os.path.join(KDAMONDS_DIR, '%d' % idx)
    _damon_sysfs.write_file(os.path.join(kd, 'state'), 'off')


def main():
    if not kcsan_available():
        print('CONFIG_KCSAN not detectably enabled '
              '(no %s); skipping -- this race has no reliable '
              'non-KCSAN oracle' % os.path.join(KCSAN_PARAMS, 'skip_watch'))
        exit(_damon_sysfs.ksft_skip)

    tune_kcsan()
    clear_dmesg()

    err = _damon_sysfs.write_file(
            os.path.join(KDAMONDS_DIR, 'nr_kdamonds'), '2')
    if err is not None:
        print('failed to create 2 kdamonds: %s' % err)
        exit(1)

    started = []
    try:
        for i in range(2):
            err = stage_kdamond(i, refresh_ms=1)
            if err is not None:
                print('kdamond.%d start failed: %s' % (i, err))
                exit(1)
            started.append(i)

        time.sleep(5)

        raced = dmesg_has_race()
        if raced is None:
            exit(1)  # couldn't read dmesg at all -- treat as failure
        if raced:
            print('BUG: %s found in dmesg -- next-refresh deadline is '
                  'shared across kdamonds' % RACE_SIGNATURE)
            exit(1)

        print('no data race observed in %s' % RACE_SIGNATURE)
        exit(0)
    finally:
        for i in started:
            stop_kdamond(i)
        _damon_sysfs.write_file(
                os.path.join(KDAMONDS_DIR, 'nr_kdamonds'), '0')


if __name__ == '__main__':
    main()
