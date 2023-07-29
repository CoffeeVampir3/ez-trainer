from tabs.train_tab import make_train_tab
from mechanisms.train import rebuild_dictionaries
interface = make_train_tab(launch_fn=rebuild_dictionaries)
interface.launch()