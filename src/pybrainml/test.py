import __init__ as bml
from __init__ import ElectrodeType, Boards
import json

exp = bml.create_experiment()

exp.metadata.subject_info.setup("Martin McCorkle", 21, "M")
exp.metadata.hardware_info.setup(ElectrodeType.HYBRID, Boards.OpenBCI_Ganglion)

with open("test.json", "w") as f:
    json.dump(exp.to_dict(), f, indent=4)