import configparser

def get_conf_section(section) : 
    config = configparser.ConfigParser()
    config.read("./config.ini")
    return config[section]

def get_conf(section, key) : 
    config = configparser.ConfigParser()
    config.read("./config.ini")
    try:
        return config[section][key]
    except:
        return 0