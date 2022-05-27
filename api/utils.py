import yaml 
import joblib


def read_config(config_path):
    with open(config_path) as cfg_file:
        content = yaml.safe_load(cfg_file)
    return content

def load_bins(scalar_path,cat_path):
    #loading cat cols
    with open(cat_path , 'rb') as handle:
        cat_dict = joblib.load(handle)
    #loading std scalar
    with open(scalar_path,'rb') as handle:
        scalar = joblib.load(handle)
    return scalar,cat_dict

