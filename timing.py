"""Per-forward timing for the MiniMax H3 DiT, recorded next to this package.

The suite's node runs before sampling starts, so it can only describe the
*configuration*. Measured speed has to come from somewhere that runs during
sampling, which is this module: it wraps ``diffusion_model._forward``, records
one sample per model call, and appends a summary line to ``h3_accel_stats.jsonl``
a few seconds after the run goes quiet.

One call is one denoiser forward. With CFG that is two calls per sampling step,
so compare ``median_ms`` between runs rather than assuming it equals step time.
"""

import json
import logging
import os
import statistics
import threading
import time

import torch

log = logging.getLogger("H3Accel")

STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h3_accel_stats.jsonl")
_FORWARD_PATH = "diffusion_model._forward"
_MARKER = "_h3_accel_timing_fallback"
_FLUSH_DELAY = 3.0

_RECORDER = None
_RECORDER_LOCK = threading.Lock()


class SectionProfiler:
    """Split one DiT forward into modulation / attention / gate / mlp.

    Events are recorded asynchronously inside the block loop and read back only
    after the forward's final device sync, so the measurement does not serialise
    the pipeline. 350 events per forward costs a few milliseconds against a
    multi-second step.
    """

    SECTIONS = ("modulation", "attention", "gate", "mlp")

    def __init__(self):
        self.enabled = False
        self.totals = {}
        self.forwards = 0
        self._pending = []

    def mark(self):
        if not self.enabled:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def add(self, section, start, end):
        if start is None or end is None:
            return
        self._pending.append((section, start, end))

    def flush(self):
        """Read back pending spans. Call only after a full device sync."""
        if not self._pending:
            return
        for section, start, end in self._pending:
            try:
                self.totals[section] = self.totals.get(section, 0.0) + start.elapsed_time(end)
            except Exception:  # noqa: BLE001 - a lost event must not break timing
                continue
        self._pending = []
        self.forwards += 1

    def snapshot(self):
        if not self.totals:
            return None
        total = sum(self.totals.values())
        if total <= 0:
            return None
        return {
            name: [round(100.0 * ms / total, 1), round(ms / max(self.forwards, 1), 2)]
            for name, ms in self.totals.items()
        }

    def reset(self):
        self.totals = {}
        self.forwards = 0
        self._pending = []


_PROFILER = SectionProfiler()


def get_profiler():
    """Shared profiler the vendored block patch writes into."""
    return _PROFILER


class RunRecorder:
    """Collect samples for one run and flush them once it stops producing."""

    def __init__(self, label, config):
        self.label = label
        self.config = config
        self.samples = []
        self._timer = None
        self._lock = threading.Lock()

    def record(self, ms):
        with self._lock:
            self.samples.append(ms)
            if self._timer is not None:
                self._timer.cancel()
            # A fixed delay shorter than one step would flush after every call,
            # turning one run into dozens of one-sample records. Scale it to the
            # step so a run lands as a single row.
            delay = max(_FLUSH_DELAY, 2.0 * ms / 1000.0)
            self._timer = threading.Timer(delay, self.flush)
            self._timer.daemon = True
            self._timer.start()

    def flush(self):
        with self._lock:
            if not self.samples:
                return
            samples, self.samples = self.samples, []
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "label": self.label or "-",
            "config": self.config,
            "calls": len(samples),
            "median_ms": round(statistics.median(samples), 2),
            "mean_ms": round(statistics.fmean(samples), 2),
            "min_ms": round(min(samples), 2),
            "max_ms": round(max(samples), 2),
        }
        sections = _PROFILER.snapshot()
        if sections:
            entry["sections"] = sections
        try:
            with open(STATS_PATH, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.info(
                "[H3 Accel] timing %s | %s | median %.2f ms over %d calls",
                entry["label"],
                entry["config"],
                entry["median_ms"],
                entry["calls"],
            )
        except Exception as exc:  # noqa: BLE001 - stats must never break a run
            log.warning("[H3 Accel] could not write timing stats: %s", exc)


def install_timing(model, label, config):
    """Wrap the DiT forward with a timer. Returns the recorder."""
    global _RECORDER

    original = model.get_model_object(_FORWARD_PATH)
    while hasattr(original, _MARKER):
        original = getattr(original, _MARKER)

    recorder = RunRecorder(label, config)

    def timed(*args, **kwargs):
        # Sync on both sides so the sample covers this call's GPU work and
        # nothing queued by the previous one. Two syncs per multi-second
        # forward are noise, and both sides of an A/B pay the same cost.
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            # Events are complete once the sync above returns, so the section
            # spans can be read back here and the pool is free for the next call.
            _PROFILER.flush()
            recorder.record((time.perf_counter() - started) * 1000.0)

    setattr(timed, _MARKER, original)

    patched = model.clone()
    patched.add_object_patch(_FORWARD_PATH, timed)
    with _RECORDER_LOCK:
        _RECORDER = recorder
    return patched, recorder


def read_records():
    """Newest-last list of recorded runs; tolerates a missing or torn file."""
    if not os.path.exists(STATS_PATH):
        return []
    out = []
    try:
        with open(STATS_PATH, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


def format_table(records, last_n=8):
    if not records:
        return (
            "还没有计时记录。\n"
            "跑一次生成，等 3 秒让它落盘，再执行这个节点。\n"
            f"记录文件：{STATS_PATH}"
        )
    rows = records[-max(1, int(last_n)):]
    baseline = rows[0]["median_ms"] if rows[0]["median_ms"] else 0.0
    lines = [
        f"{'时间':<20}{'标签':<12}{'每次前向 ms':>12}{'相对首行':>10}  配置",
        "-" * 96,
    ]
    for entry in rows:
        median = entry.get("median_ms", 0.0) or 0.0
        ratio = f"{baseline / median:.3f}x" if median else "-"
        lines.append(
            f"{entry.get('time',''):<20}"
            f"{entry.get('label','-'):<12}"
            f"{median:>12.2f}"
            f"{ratio:>10}  "
            f"{entry.get('config','')}"
        )
    lines.append("-" * 96)
    lines.append("相对首行 > 1 表示比首行快。同一配置多跑几次取中位数更稳。")

    latest = rows[-1]
    sections = latest.get("sections")
    if sections:
        names = {"modulation": "归一化+调制", "attention": "注意力", "gate": "门控残差", "mlp": "前馈"}
        lines.append("")
        lines.append(f"最近一次分段占比（{latest.get('label','-')}，每次前向均值）：")
        for key in ("attention", "mlp", "modulation", "gate"):
            if key in sections:
                pct, ms = sections[key]
                lines.append(f"  {names.get(key, key):<14}{pct:>6.1f}%{ms:>10.2f} ms")
    else:
        lines.append("")
        lines.append("没有分段数据：需要在 Accel Suite 节点上打开 profile_sections。")

    lines.append(f"完整记录：{STATS_PATH}")
    return "\n".join(lines)


def image_diff(image_a, image_b):
    """Compare two IMAGE batches ([B,H,W,C] in 0..1) and describe the difference."""
    import math

    if image_a is None or image_b is None:
        return None
    a = image_a[:1, ..., :3].float()
    b = image_b[:1, ..., :3].float()
    if a.shape != b.shape:
        return f"图像尺寸不同：{tuple(a.shape)} vs {tuple(b.shape)}，无法逐像素比较"
    delta = (a - b).abs()
    mse = float((delta ** 2).mean())
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
    same = bool(torch.equal(a, b))
    return (
        f"图像差异：平均 {float(delta.mean()):.6f} · 最大 {float(delta.max()):.6f} · "
        f"PSNR {psnr:.2f} dB · {'完全一致' if same else '存在差异'}"
    )
