


class Config(object):

    def __init__(self, name):
        self.name = name
    
    def initialize(self, args):
        """initialize configuration by args dictionary

        Args:
            args (dict): 
        """
        for key, value in args.items():
            setattr(self, key, value)