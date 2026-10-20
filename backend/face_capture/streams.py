"""Owner-scoped fresh monitoring frames, with bounded persistent fallback."""
import threading
import cv2

_context = None
_lock = threading.RLock()
_streams = {}
_cursors = {}
MAX_STREAMS = 4


class FramePending(RuntimeError):
    pass


def bind_live_context(provider):
    global _context
    _context = provider


def fresh(frame, health):
    age = (health or {}).get('frame_age_ms')
    return frame is not None and age is not None and 0 <= age <= 500 and health.get('status') == 'live'


def claim_frame(key, capture, sequence):
    with _lock:
        cursor = (capture, sequence)
        if _cursors.get(key) == cursor:
            raise FramePending('等待下一帧')
        _cursors[key] = cursor


def shared_frame(owner_id, recorder_id, channel_number, url):
    if _context is None:
        return None
    processor, contexts = _context()
    source = f'recorder:{recorder_id}:channel:{channel_number}'
    for session_id, context in list(contexts.items()):
        if str(context.get('owner_id')) != str(owner_id) or context.get('source_id') != source:
            continue
        with processor._capture_lock:
            capture = processor.supervised_captures.get(session_id)
        if capture is None or not capture.matches('rtsp', url):
            continue
        image, sequence, health = capture.wait_for_latest(0, 0)
        if fresh(image, health):
            claim_frame((str(owner_id), recorder_id, channel_number), capture, sequence)
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return None


def _open_capture(_kind, url):
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000])
        if cap.isOpened():
            return cap
        cap.release()
    except Exception:
        pass
    return None


def _read_frame(cap, _rgb):
    ok, frame = cap.read()
    return frame if ok else None


def close_camera(key):
    with _lock:
        stream = _streams.get(key)
        if stream is not None and stream.stop(join_timeout=.05):
            _streams.pop(key, None)


def retain_cameras(keys):
    with _lock:
        obsolete = [key for key in _streams if key not in keys]
        for key in list(_cursors):
            if key not in keys:
                _cursors.pop(key, None)
    for key in obsolete:
        close_camera(key)


def shutdown():
    with _lock:
        streams = list(_streams.values())
        _streams.clear()
        _cursors.clear()
    for stream in streams:
        stream.stop(join_timeout=4)


def latest_frame(owner_id, recorder_id, channel_number, url):
    from utils.video_utils import SupervisedCapture
    key = (str(owner_id), recorder_id, channel_number)
    image = shared_frame(*key, url)
    if image is not None:
        close_camera(key)
        return image
    with _lock:
        stream = _streams.get(key)
        if stream is not None and (not stream.matches('rtsp', url) or stream.snapshot()['status'] in ('stopped', 'stopping')):
            close_camera(key)
            if key in _streams:
                raise FramePending('正在切换视频连接')
            stream = None
        if stream is None:
            if len(_streams) >= MAX_STREAMS:
                raise RuntimeError('独立人脸视频连接已达上限，请先启用对应监控视频流')
            stream = SupervisedCapture('face-' + recorder_id + '-' + channel_number, 'rtsp', url,
                                       _open_capture, _read_frame, lambda cap: cap.release(),
                                       reconnect_initial_seconds=.2, reconnect_max_seconds=3).start()
            _streams[key] = stream
    image, sequence, health = stream.wait_for_latest(0, .02)
    if not fresh(image, health):
        raise FramePending('等待摄像头最新画面')
    claim_frame(key, stream, sequence)
    return image.copy()
