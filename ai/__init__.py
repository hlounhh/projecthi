"""Image posture detection. No temporal fall verification or pose keypoints yet.

Integration contract: consume a new FramePacket from reader.snapshot(); use
received_at for local intervals, sequence for skipped frames, and never mutate
packet.image. Long-running inference must not block the capture worker.
"""
