import sys
from colorama import Fore, init
init(autoreset=True)

def ErrorProcessor(code, file, linenumber, type=None, tracebacklimit=0):
    sys.tracebacklimit = tracebacklimit
    if type == "mySql":
        raise ValueError(f"Status Code = {code.errno} |{str(str(code).split(':', 1)[1])} | In {file}:{linenumber}")

    # Mysql Errors
    if code == 200:
        print(f"Status Code = {code} | Request completed successfully | In {file}:{linenumber}")
        pass
    elif code == 201:
        raise ValueError(f"Status Code = {code} | No data found with given acctrefno | In {file}:{linenumber}")
    elif code == 901:
        raise ValueError(f"Status Code = {code} | No valid Selection made | In {file}:{linenumber}")
    elif code == 1001:
        print(Fore.RED + "\nFile Not Found ERROR: " + Fore.YELLOW + "Please follow the instructions regarding file placement above.", flush=True)
    # Catch All
    else:
        raise Exception(f"Status Code = {code} | An unknown exception occurred | In {file}:{linenumber}")
