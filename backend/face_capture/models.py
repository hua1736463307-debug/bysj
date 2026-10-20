from sqlalchemy import Boolean, Column, Float, Integer, JSON, String, Text
from sqlalchemy.orm import declarative_base
import time

Base = declarative_base()


class FrameSample(Base):
    """A durable signal: one native alarm frame plus its same-frame person ROIs."""
    __tablename__ = 'face_frame_samples'
    id = Column(String(64), primary_key=True)
    owner_id = Column(String(64), nullable=False, index=True)
    source_id = Column(String(64), nullable=False, index=True)
    monitor_id = Column(String(36), nullable=False)
    run_id = Column(String(64), nullable=False)
    source_sequence = Column(String(64), nullable=False)
    alarm_frame_index = Column(String(64), nullable=False)
    frame_key = Column(String(256), nullable=False)
    camera_id = Column(String(160), nullable=False)
    camera_name = Column(String(200), nullable=False)
    captured_at = Column(String(40), nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    person_boxes = Column(JSON, nullable=False)
    settings_json = Column(JSON, nullable=False)
    image_sha256 = Column(String(64), nullable=False)
    byte_count = Column(Integer, nullable=False)
    status = Column(String(24), nullable=False, default='pending', index=True)
    attempts = Column(Integer, nullable=False, default=0)
    lease = Column(String(40), nullable=True)
    lease_until = Column(Float, nullable=False, default=0, index=True)
    created_at = Column(Float, nullable=False, default=time.time)
    expires_at = Column(Float, nullable=False)
    completed_at = Column(Float, nullable=True)
    face_count = Column(Integer, nullable=False, default=0)
    cache_released = Column(Boolean, nullable=False, default=False)
    last_error = Column(Text, nullable=True)


class Capture(Base):
    __tablename__ = 'face_captures'
    id = Column(String(64), primary_key=True)
    owner_id = Column(String(64), nullable=False, index=True)
    camera_id = Column(String(160), nullable=False)
    camera_name = Column(String(200), nullable=False)
    frame_id = Column(String(100), nullable=False)
    captured_at = Column(String(40), nullable=False)
    bbox = Column(JSON, nullable=False)
    confidence = Column(Float, nullable=False)
    model_sha256 = Column(String(64), nullable=False)
    image_sha256 = Column(String(64), nullable=False)
    status = Column(String(20), nullable=False, index=True)
    destination = Column(String(20), nullable=False, default='test')
    attempts = Column(Integer, nullable=False, default=0)
    next_at = Column(Float, nullable=False, default=0, index=True)
    lease = Column(String(40), nullable=True)
    last_error = Column(Text, nullable=True)
    delivered_at = Column(String(40), nullable=True)


class Source(Base):
    __tablename__ = 'face_capture_sources'
    id = Column(String(64), primary_key=True)
    owner_id = Column(String(64), nullable=False, index=True)
    recorder_id = Column(String(64), nullable=False)
    channel_number = Column(String(64), nullable=False)
    enabled = Column(Boolean, nullable=False, default=False)
    push_enabled = Column(Boolean, nullable=False, default=False)
    interval_seconds = Column(Float, nullable=False, default=2)
    threshold = Column(Float, nullable=False, default=.8)
    min_size = Column(Integer, nullable=False, default=20)
    next_at = Column(Float, nullable=False, default=0)
    lease = Column(String(40), nullable=True)
    lease_until = Column(Float, nullable=False, default=0)
    last_error = Column(Text, nullable=True)
    last_at = Column(String(40), nullable=True)
    last_count = Column(Integer, nullable=False, default=0)


class Receipt(Base):
    __tablename__ = 'face_test_receipts'
    id = Column(String(64), primary_key=True)
    owner_id = Column(String(64), nullable=False, index=True)
    camera_name = Column(String(200), nullable=False)
    camera_id = Column(String(160), nullable=False)
    captured_at = Column(String(40), nullable=False)
    received_at = Column(String(40), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    image_sha256 = Column(String(64), nullable=False)
