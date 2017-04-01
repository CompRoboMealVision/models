from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from inception.dataset import Dataset

class FoodsData(Dataset):
  """Foods data set."""

  def __init__(self, subset):
    super(FoodsData, self).__init__('Foods', subset)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 101

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data subset."""
    if self.subset == 'train':
      return 80800
    if self.subset == 'validation':
      return 20200

  def download_message(self):
      pass
