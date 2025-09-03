from typing import Dict

EPS = 1e-24


class BaseLogger:
    def log(self, *args, **kwargs):
        raise NotImplementedError

    def define_metric(self, *args, **kwargs):
        raise NotImplementedError

    def process_image(self, image):
        return image

    def process_video(self, video):
        return video

    def process_figure(self, figure):
        return figure

    def process_dist(self, dist):
        return dist

    @property
    def name(self):
        raise NotImplementedError

class BaseMetric:
    logger: BaseLogger
    def __init__(self, logger: BaseLogger, runner,
                 update_step, log_step, update_period, log_period):
        self.logger = logger
        self.runner = runner
        self.update_step = update_step
        self.log_step = log_step
        self.update_period = update_period
        self.log_period = log_period

        self.last_update_step = None
        self.last_log_step = None

    def step(self):
        update_step = self.get_attr(self.update_step)
        log_step = self.get_attr(self.log_step)

        if (self.last_update_step is None) or (self.last_update_step != update_step):
            if (update_step % self.update_period) == 0:
                self.update()

        if (self.last_log_step is None) or (self.last_log_step != log_step):
            if (log_step % self.log_period) == 0:
                self.log(log_step)

        self.last_update_step = update_step
        self.last_log_step = log_step

    def update(self):
        raise NotImplementedError

    def log(self, step):
        raise NotImplementedError

    def get_attr(self, attr):
        obj = self.runner
        for a in attr.split('.'):
            obj = getattr(obj, a)
        return obj


class MetricsRack:
    metrics: Dict[str, BaseMetric]

    def __init__(self, logger, runner, **kwargs):
        self.metrics = dict()

        for name, params in kwargs.items():
            cls = params['class']
            params = params['params']
            self.metrics[name] = eval(cls)(**params, logger=logger, runner=runner)

    def step(self):
        for name in self.metrics.keys():
            self.metrics[name].step()