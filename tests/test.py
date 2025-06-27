import pybrainml as bml
from pybrainml import ElectrodeType, Boards, Frame
import json
from datetime import datetime
import os

# Setup
port = "COM8"

exp = bml.create_experiment()

exp.metadata.subject_info.setup("Martin McCorkle", 21, "M")
exp.metadata.hardware_info.setup(ElectrodeType.HYBRID, Boards.OpenBCI_Ganglion)


# Run Experiment
frame = Frame.create("NA", datetime.now().isoformat(), [])

buffer = bml.start_eeg_stream(port)

frame.eeg_data = list(buffer)
exp.frames.append(frame)

with open("test.json", "w") as f:
    json.dump(exp.to_dict(), f, indent=4)