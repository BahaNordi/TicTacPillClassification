import os
from tensorboardX import SummaryWriter


def create_summary_writer(model, data_loader, log_dir) -> SummaryWriter:
    if os.path.isdir(log_dir):
        print('WARNING: overwrite already existing tensorboard dir {}'.format(log_dir))
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


class TensorBoardHandler(object):
    def __init__(self, writer: SummaryWriter, data_loader, prefix: str = None):
        self.writer = writer
        self.prefix = prefix
        self.epoch_len = len(data_loader)
        self.iteration = 0
        self.epoch = 0
        self.step = 0

    def write_tb(self, metric, epoch=None, iteration=None, per_iteration=True):
        self.epoch = epoch
        self.iteration = iteration
        if iteration is not None:
            self._calculate_step()
        for metric_name, val in metric.items():
            scalar_name = self.prefix + '/' + metric_name if self.prefix else metric_name
            if per_iteration:
                self.writer.add_scalar(scalar_name, val, self.step)
            else:
                self.writer.add_scalar(scalar_name, val, self.epoch)

    def _calculate_step(self):
        self.step = self.iteration + self.epoch*self.epoch_len
