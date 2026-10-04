"""Read-only down-segment diagnostics; never an acceptance reference replacement."""
import numpy as np


def decode_trace(raw, width, segments, cores=8):
    if width <= 0 or segments <= 0 or cores <= 0 or width % (cores*64):
        raise ValueError('trace size/geometry')
    if raw.size != segments*4*width:
        raise ValueError('trace size')
    if not np.isfinite(raw).all():
        raise ValueError('nonfinite trace')
    return raw.reshape(cores, segments, width//cores//64, 4, 64).transpose(
        1, 3, 0, 2, 4).reshape(segments, 4, width)


def analyze(trace, parts, channel):
    if trace.shape != (len(parts), 4, parts.shape[1]):
        raise ValueError('trace/reference shape')
    if not 0 <= channel < parts.shape[1]:
        raise ValueError('channel out of range')
    if not np.isfinite(trace).all() or not np.isfinite(parts).all():
        raise ValueError('nonfinite input')
    t = trace[:, :, channel].astype(np.float64)
    device_parts = t[:, 0] + t[:, 1]
    accumulated = t[:, 2] + t[:, 3]
    expected = parts[:, channel]
    return dict(diagnostic_only=True, channel=channel,
                segment_high=t[:, 0].tolist(), segment_low=t[:, 1].tolist(),
                accumulated_high=t[:, 2].tolist(), accumulated_low=t[:, 3].tolist(),
                reference_parts=expected.tolist(),
                segment_error=(device_parts-expected).tolist(),
                reduction_error=(accumulated-device_parts.cumsum()).tolist(),
                fp64_reduce_device_parts=float(np.float32(device_parts.sum())),
                reference_final=float(np.float32(expected.sum())),
                device_final=float(np.float32(accumulated[-1])))
