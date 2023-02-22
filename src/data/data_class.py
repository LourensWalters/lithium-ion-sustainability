from dataclasses import dataclass
import pandas as pd

# TODO: Should look into using private attributes and using getters and setters
@dataclass
class BatteryData:
    name: object
    data_dict: dict             # all_batches_dict
    data_frame: pd.DataFrame    # read from SQL or pickle
