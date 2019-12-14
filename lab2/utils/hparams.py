import sys
import inspect
import json


class HyperParams:
    """
    A class for placing hyper parameters for training.
    Expected to display during training and save down to experiement / train folder
    """
    discount_factor = 0.95
    learning_rate = 0.005
    memory_size = 1000
    target_update_frequency = 1


    def __init__(self, **kwargs):
        for k in kwargs.keys():
            if hasattr(self, k):
                setattr(self, k, kwargs[k])

    @classmethod
    def from_dict(cls, hparams):
        return cls(**hparams)

    def to_dict(self):
        return self._get_param_dict()

    def _get_param_dict(self):
        ret = dict()
        for (attr, val) in inspect.getmembers(self):
            # Ignores anything starting with underscore
            # (that is, private and protected attributes)
            if not attr.startswith('_'):
                # Ignores methods
                if not inspect.ismethod(val):
                    ret[attr] = val
        return ret

    def __repr__(self):
        params = self._get_param_dict()
        return repr(params)

    def display(self):
        self.print_on(file=sys.stdout)

    def print_on(self, file):
        params = self._get_param_dict()
        print("--------- Hyper Parameters ---------", file=file)
        for k, v in params.items():
            print("{:30} {}".format(k, v), file=file)
        print("--------- Hyper Parameters ---------", file=file)

    def save_to_txt(self, fname):
        with open(fname, 'w') as f:
            self.print_on(f)
            f.write('\n')

    def save_to_json(self, fname):
        with open(fname, 'w') as f:
            json.dump(self._get_param_dict(), f, indent=2)
            f.write('\n')
