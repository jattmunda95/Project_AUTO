"""Planned stage-blind crop preparation and identity comparison; not implemented."""

# TODO(identity): Accept a coordinator request after tracker confirmation, not tracker state.
# Crop the detection, pass crop-local boxes to SAM, apply the inverted keep-mask
# (True retains pixels), and embed with the same preprocessing as gallery references.
# Return NEW / EXISTING / PENDING plus source_track_id, item_id, similarity, and reusable
# crop/embedding evidence. An unusable observation is PENDING, never automatically NEW.
# The coordinator owns retries, retirement, gallery refresh, and six-reference scheduling.
# The event/store layer owns item creation, status checks, reference writes, and prototypes.
# Keep this module disconnected until standalone tests and real-image checks pass.
