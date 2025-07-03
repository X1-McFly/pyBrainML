from brainflow.board_shim import BoardShim, BrainFlowInputParams
from yaspin import yaspin
import os
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

BoardShim.disable_board_logger() 

def connect_board(port: str, board_id: int, max_retries: int):

    params = BrainFlowInputParams()
    params.serial_port = port   

    def attempt_connect():
        board = BoardShim(board_id, params)
        try:
            board.prepare_session()
            board.release_session()
            return True
        except Exception:
            return False

    os.system('cls' if os.name == 'nt' else 'clear')
    with yaspin(text="Connecting to board...", color="green") as spinner:
        for attempt in range(1, max_retries + 1):
            if attempt_connect():
                spinner.text = ""
                spinner.ok("Connected")
                break
            elif attempt < max_retries:
                spinner.text = f"Attempt {attempt}/{max_retries} failed..."
                spinner.color = "yellow"
            else:
                spinner.text = ""
                spinner.color = "red"
                spinner.fail("Connection failed")
