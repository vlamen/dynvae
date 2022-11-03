from collections import UserDict
import pandas as pd


class Settings(UserDict):
    """
    Settings a wrapper around dictionary that allows object-attribute like access so I don't have to mess with strings all the time
    """
    _initialized = False

    def __init__(self, initial_data=None, /, **kwargs):
        super().__init__(initial_data, **kwargs)
        self._initialized = True

    def __getattr__(self, attr):
        if attr == 'data':
            # this might happen in loading with pickle
            raise AttributeError(attr)
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(attr)

    def __setattr__(self, attr, value):
        if self._initialized:
            self[attr] = value
        else:
            object.__setattr__(self, attr, value)

    def __hasattr__(self, attr):
        return object.__hasattr__(self, attr) or attr in self

    def __delattr__(self, name):
        if not self.__hasattr__(name):
            raise AttributeError(f'Cannot delete attribute with name {name} because no such attribute exists')
        if name not in self.data:
            raise AttributeError(f'Cannot delete attribute with name {name} because it is not an item.')
        del self.data[name]


def format_tree(tree, indent=4, config_name='config'):
    lines = []  # will be the list of all lines to be returned
    white_space = indent * ' '
    if isinstance(tree, (dict, UserDict)):
        lines.append('{')
        for key in tree:
            key_content = format_tree(tree[key])
            lines.append(f'{white_space}"{key}": {key_content.pop(0)}')
            for line in key_content[:-1]:
                lines.append(white_space+line)
            if key_content:
                lines.append(white_space+key_content[-1]+',')
        lines.append('}')
        return lines
    elif isinstance(tree, (list, tuple)):
        lines.append('[') # tuples become lists
        for sub_tree in tree:
            key_content = format_tree(sub_tree)
            lines.append(white_space + key_content.pop(0))
            for line in key_content[:-1]:
                lines.append(white_space + line)
            if key_content:
                lines.append(white_space + key_content[-1] + ',')
        lines.append(']')
    elif hasattr(tree, '__name__'):
        if hasattr(tree, '_arguments'):
            lines.append(f'{config_name}["{tree.__name__}"]({",".join(str(arg) for arg in tree._arguments)}),')
        else:
            lines.append(f'{config_name}["{tree.__name__}"],')
    elif isinstance(tree, str):
        lines.append(f'"{tree}",')
    else:
        lines.append(f'{str(tree)},')
    return lines


def parse_settings(string_representation, config, config_name='config'):
    dict_representation = eval(string_representation, {config_name: config})
    return Settings(dict_representation)


def load_settings(path, config, config_name='config'):
    with open(path, 'r') as settings_file:
        content = '\n'.join(line for line in settings_file)
    return parse_settings(content, config=config, config_name=config_name)
    

def format_field(value, config_name='config'):
    if hasattr(value, '__name__'):
        if hasattr(value, '_arguments'):
            return f'{config_name}["{value.__name__}"]({",".join(str(arg) for arg in value._arguments)})'
        else:
            return f'{config_name}["{value.__name__}"]'
    else:
        return str(value)


def get_difference_table(bunch_of_settings, exclude=('profiles',)):
    all_settings_fields = set()
    all_settings_fields = all_settings_fields.union(
        *tuple(set(settings.keys()) for settings in bunch_of_settings.values()))

    varying_fields = []
    for field in all_settings_fields:
        if field in exclude:
            continue  # skip this one
        refference_settings = next(iter(bunch_of_settings.values()))
        if any(settings.get(field, None) != refference_settings.get(field, None) for settings in
               bunch_of_settings.values()):
            varying_fields.append(field)
    print(varying_fields)
    return pd.DataFrame.from_dict(
        data={exp_id: [format_field(settings.get(field, None)) for field in varying_fields] for exp_id, settings in
              bunch_of_settings.items()},
        orient='index',
        columns=varying_fields
    )
