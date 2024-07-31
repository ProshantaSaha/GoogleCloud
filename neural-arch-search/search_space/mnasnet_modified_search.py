import pyglove as pg
from six.moves import zip

from architectures import mnasnet_defaults

MOBILENET_V2_NUM_REPEATS = [1, 2, 3, 4, 3, 3, 1]
MOBILENET_V2_OUTPUT_FILTERS = [16, 24, 32, 64, 96, 160, 320]
MNASNET_A1_NUM_REPEATS = [1, 2, 3, 4, 2, 3, 1]
MNASNET_A1_OUTPUT_FILTERS = [16, 24, 40, 80, 112, 160, 320]

@pg.members([
    ('num_repeats', pg.typing.Int(default=1),
     'The number of repeats for each block.'),
    ('block_fn', pg.typing.Enum(default='mbconv', values=['conv', 'mbconv', 'fused_mbconv']),
     'The type of block functions.'),
    ('expand_ratio', pg.typing.Int(default=1),
     'The expansion ratio of the MBConv block.'),
    ('kernel_size', pg.typing.Int(default=3), 'The kernel size.'),
    ('se_ratio', pg.typing.Float(default=0.0), 'The squeeze-excitation ratio.'),
    ('output_filters', pg.typing.Int(), 'The number of output filters'),
])
class TunableBlockSpec(pg.Object):
  """The tunable specifications of a MnasNet block."""


@pg.functor([
    ('blocks',
     pg.typing.List(
         pg.typing.Dict([
             ('num_repeats', pg.typing.Int(default=1),
              'The number of repeats for each block.'),
             ('block_fn',
              pg.typing.Enum(default='mbconv', values=['conv', 'mbconv', 'fused_mbconv']),
              'The type of block functions.'),
             ('expand_ratio', pg.typing.Int(default=1),
              'The expansion ratio of the MBConv block.'),
             ('kernel_size', pg.typing.Int(default=3), 'The kernel size.'),
             ('se_ratio', pg.typing.Float(default=0.0),
              'The squeeze-excitatio ratio.'),
             ('output_filters', pg.typing.Int(),
              'The number of output filters'),
         ]), size=mnasnet_defaults.MNASNET_NUM_BLOCKS)),
])
def build_tunable_block_specs(blocks):
  """Builds the MnasNet block specification."""

  # pylint: disable=g-complex-comprehension
  return [
      TunableBlockSpec(
          num_repeats=b.num_repeats,
          block_fn=b.block_fn,
          expand_ratio=b.expand_ratio,
          kernel_size=b.kernel_size,
          se_ratio=b.se_ratio,
          output_filters=b.output_filters)
      for b in blocks]
  # pylint: enable=g-complex-comprehension

def mnasnet_search_space(reference='mobilenet_v2'):
  """Builds the MnasNet search space.

  Args:
    reference: supports `mobilenet_v2` and `mnasnet_a1`.

  Returns:
    a BlockSpec for MnasNet architecture.
  """
  if reference == 'mobilenet_v2':
    base_num_repeats = MOBILENET_V2_NUM_REPEATS
    base_num_output_filters = MOBILENET_V2_OUTPUT_FILTERS
  elif reference == 'mnasnet_a1':
    base_num_repeats = MNASNET_A1_NUM_REPEATS
    base_num_output_filters = MNASNET_A1_OUTPUT_FILTERS

  blocks = [
      pg.Dict(
          num_repeats=pg.one_of([r, r - 1, r + 1] if r > 1 else [1, 2]),
          block_fn=pg.one_of(['conv', 'mbconv', 'fused_mbconv']),
          expand_ratio=pg.one_of([1, 3, 6]),
          kernel_size=pg.one_of([3, 5]),
          se_ratio=pg.one_of([0.0, 0.10, 0.25, 0.50, 0.75, 1.0]),
          output_filters=pg.one_of([int(o * 0.75), o, int(o * 1.25)]))
      for r, o in zip(base_num_repeats, base_num_output_filters)]
  return build_tunable_block_specs(blocks=blocks)