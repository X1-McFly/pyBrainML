import __init__ as bml
from __init__ import ElectrodeType, Boards, Frame
import json
from datetime import datetime
# from dataclasses import asdict

exp = bml.create_experiment()

exp.metadata.subject_info.setup("Martin McCorkle", 21, "M")
exp.metadata.hardware_info.setup(ElectrodeType.HYBRID, Boards.OpenBCI_Ganglion)

frame1 = Frame.create("NA", datetime.now().isoformat(), [])

port = "COM8"
buffer = bml.start_eeg_stream(port, save_to="eeg.ndjson")

frame1.eeg_data = list(buffer)

exp.frames.append(frame1)

with open("test.json", "w") as f:
    json.dump(exp.to_dict(), f, indent=4)

if __name__ == '__main__':
    import __init__ as bml
    buffer = bml.start_eeg_stream(port, save_to="eeg.ndjson")