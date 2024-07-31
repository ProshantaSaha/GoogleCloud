MNASNET_NUM_BLOCKS = 7
MNASNET_STRIDES = [1, 2, 2, 2, 1, 2, 1]  # Same as MobileNet-V2.

# The fixed MnasNet-A1 architecture discovered by NAS.
# Each element represents a specification of a building block:
# (num_repeats, block_fn, expand_ratio, kernel_size, se_ratio, output_filters)
MNASNET_A1_BLOCK_SPECS = [
    (1, 'mbconv', 1, 3, 0.0, 16),
    (2, 'mbconv', 6, 3, 0.0, 24),
    (3, 'mbconv', 3, 5, 0.25, 40),
    (4, 'mbconv', 6, 3, 0.0, 80),
    (2, 'mbconv', 6, 3, 0.25, 112),
    (3, 'mbconv', 6, 5, 0.25, 160),
    (1, 'mbconv', 6, 3, 0.0, 320),
]