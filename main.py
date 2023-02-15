import sys
import traceback

from colorama import Fore, init

from PaymentPredictorUtility.Classes.GraphicalUserInterFace import GraphicalUserInterface
from PaymentPredictorUtility.Functions.ErrorCodes import ErrorProcessor

init(autoreset=True, convert=True)

try:
    import pyi_splash
    pyi_splash.close()
except:
    pass


if __name__ == '__main__':
    graphObj = GraphicalUserInterface()
    try:
        graphObj.showGui()
        print()
        input(Fore.GREEN + "Press Enter to Exit Program...")
    except FileNotFoundError as e:
        print(e)  # DEBUG
        graphObj.loadingAnimator.stop(method="none")
        ErrorProcessor(1001, "", "")
        input(Fore.RED + "An Error Occurred | Press Enter to Kill Program...")
    except BaseException as e:
        graphObj.loadingAnimator.stop(method="none")
        print(sys.exc_info())
        print(traceback.print_last())
        input(Fore.RED + "An Error Occurred | Press Enter to Kill Program...")
