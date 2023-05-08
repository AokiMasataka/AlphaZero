def load_config_file(path: str) -> dict:
    with open(path, 'r') as f:
        text = f.read()
    
    config = dict()
    exec(text, globals(), config)

    if '_base_' in config.keys():
        for base_config_path in config.pop('_base_'):
            base_config, _ = load_config_file(path=base_config_path)
            config = merge_configs(config=config, base_config=base_config)

    return config


def merge_configs(config: dict, base_config: dict) -> dict:
    if isinstance(base_config, dict):
        for key in base_config.keys():
            if key in config.keys():
                # print(config[key])
                config[key] = merge_configs(config=config[key], base_config=base_config[key])
            else:
                config[key] = base_config[key]
    else:
        return config
    return config


def config_encoder(config: dict, indent: int = 0) -> str:
    text = ''
    for key, value in config.items():
        if isinstance(value, dict):
            text += '\t' * indent + f'{key}=' + 'dict(' + '\n'
            indent += 1
            text += config_encoder(config=value, indent=indent)
            indent -= 1
            text += '\t' * indent + ')' + ',\n'
        elif isinstance(value, list):
            text += '\t' * indent + f'{key}=' + '[' + '\n'
            indent += 1
            for item in value:
                indent += 1
                text += '\t' * indent + str(item) + ',\n'
                indent -= 1
            indent -= 1
            text += '\t' * indent + ']' + ',\n'

        else:
            if isinstance(value, str):
                value = "'" + value + "'"
            else:
                value = str(value)
            text += '\t' * indent + f'{key}=' + value + ',\n'
    return text
