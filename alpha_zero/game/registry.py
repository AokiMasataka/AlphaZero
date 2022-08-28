__all__ = ['GAMES']


class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def register_module(self, module):
        self._module_dict[module.__name__] = module
        return module

    def get_module(self, name):
        return self._module_dict[name]


GAMES = Registry(name='games')
