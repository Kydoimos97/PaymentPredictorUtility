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
from colorama import Fore, init
init(autoreset=True)

def ErrorProcessor(code, file, linenumber, type=None, tracebacklimit=0):
    """
The ErrorProcessor function is used to raise exceptions based on the error code passed into it.
    The function takes in 5 arguments:
        1) code - This is the error code that will be used to determine what exception should be raised.
        2) file - This is a string containing the name of the file where this function was called from.  It's purpose is for debugging purposes only, and can be removed if desired.
        3) linenumber - This integer contains which line number this function was called from within its calling script/module/file, and like 'file' above, it's purpose is for debugging purposes

Args:
    code: Determine what error to raise
    file: Determine where the error occurred
    linenumber: Display the line number in which the error occurred
    type: Specify the type of error that is being processed
    tracebacklimit: Limit the amount of traceback information that is returned

Returns:
    A string

Doc Author:
    Willem van der Schans, Trelent AI
"""
    sys.tracebacklimit = tracebacklimit
    if type == "mySql":
        raise ValueError(f"Status Code = {code.errno} |{str(str(code).split(':', 1)[1])} | In {file}:{linenumber}")

   
    if code == 200:
        print(f"Status Code = {code} | Request completed successfully | In {file}:{linenumber}")
        pass
    elif code == 201:
        raise ValueError(f"Status Code = {code} | No data found with given acctrefno | In {file}:{linenumber}")
    elif code == 901:
        raise ValueError(f"Status Code = {code} | No valid Selection made | In {file}:{linenumber}")
    elif code == 1001:
        print(Fore.RED + "\nFile Not Found ERROR: " + Fore.YELLOW + "Please follow the instructions regarding file placement above.", flush=True)
   
    else:
        raise Exception(f"Status Code = {code} | An unknown exception occurred | In {file}:{linenumber}")
