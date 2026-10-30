Preserve small rigid-body velocities in XPBD instead of zeroing them after
each constraint correction. The implicit cutoff could stall slowly driven
mechanisms as their substep count increased and erase velocity gradients.
