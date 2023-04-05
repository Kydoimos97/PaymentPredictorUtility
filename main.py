#  Copyright (C) 2022-2023 - Willem van der Schans - All Rights Reserved.
#
#  THE CONTENTS OF THIS PROJECT ARE PROPRIETARY AND CONFIDENTIAL.
#  UNAUTHORIZED COPYING, TRANSFERRING OR REPRODUCTION OF THE CONTENTS OF THIS PROJECT, VIA ANY MEDIUM IS STRICTLY PROHIBITED.
#  The receipt or possession of the source code and/or any parts thereof does not convey or imply any right to use them
#  for any purpose other than the purpose for which they were provided to you.
#
#  The software is provided "AS IS", without warranty of any kind, express or implied, including but not limited to
#  the warranties of merchantability, fitness for a particular purpose and non infringement.
#  In no event shall the authors or copyright holders be liable for any claim, damages or other liability,
#  whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software
#  or the use or other dealings in the software.
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

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
