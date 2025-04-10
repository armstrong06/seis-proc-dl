import json

"""Copied this class from https://github.com/The-AI-Summer/Deep-Learning-In-Production"""


class Config:
    """Config class which contains data, train, and model hyperparameters"""

    def __init__(self, unet, paths, dataloader, database=None):
        self.unet = unet
        self.paths = paths
        self.dataloader = dataloader
        self.database = database

    @classmethod
    def from_json(cls, cfg):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        if "database" in params.__dict__.keys():
            init = cls(params.unet, params.paths, params.dataloader, params.database)
        else:
            init = cls(params.unet, params.paths, params.dataloader)
        return init


class HelperObject(object):
    """Helper class to convert json into Python object"""

    def __init__(self, dict_):
        self.__dict__.update(dict_)
