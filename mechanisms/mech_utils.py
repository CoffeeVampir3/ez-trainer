import os

def get_path_from_leaf(root, leaf):
    return os.path.join(root, leaf)

def pick_devices(device_selection):
    devices = "auto"
    if device_selection is None or device_selection == 0:
        devices = "cpu"
    return devices