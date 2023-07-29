import os, sys

def find_leaf_directories(parent_directory):
    leaf_directories = []
    for item in os.scandir(parent_directory):
        # If it's a directory, check if it has any visible subdirectories
        if item.is_dir():
            if all(name.startswith('.') or not os.path.isdir(os.path.join(item.path, name)) for name in os.listdir(item.path)):
                leaf_directories.append(item.path)
    return leaf_directories

def find_leaf_files(parent_directory):
    files = []
    for item in os.scandir(parent_directory):
        # If it's a file, add it to the list
        if item.is_file():
            files.append(item.path)
    return files
    
def get_path_from_leaf(root, leaf):
    return os.path.join(root, leaf)

def get_available_from_dir(target):
    return [os.path.basename(x) for x in find_leaf_directories(target)]

def get_available_from_leafs(target):
    return [os.path.basename(x) for x in find_leaf_files("datasets")]