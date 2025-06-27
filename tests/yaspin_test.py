from yaspin import yaspin
import time
from rich import print

try:
    with yaspin(text="Working...", color="green") as spinner:
        while True:
            time.sleep(0.01)
except KeyboardInterrupt:
    print("\n[red][!][/] Interrupted. Exiting cleanly.")
