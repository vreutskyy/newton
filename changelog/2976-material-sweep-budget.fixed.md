Honor `tendon_max_sweeps` values above 256 in XPBD and VBD material solves.
The default budget remains 256; larger explicit budgets can resolve long,
stiff tendon routes that previously stopped at an internal hard limit.
Recognize convergence when a span's minimum rest length prevents further
material transfer, while still allowing transfer away from that bound.
