#=========================================================================
# For Debugging :
# we use from behold import Behold as dbg and some macros defined as follows
# 2020 07 02 by ***********
#=========================================================================
import inspect
import sys
import os
# If you want to represent debug message, set the "_active" to be True
def dbg(*msg, _active=True):
    if _active :
        _file_name = inspect.stack()[1][1]     ## Full length of file name, so that we don't use this parameter
        _line_no    = inspect.stack()[1][2]
        _func_name  = inspect.stack()[1][3]
        _str        = ''
        _prev_int   = False
        for _msg_str in msg:
            if str(type(_msg_str)) == "<class 'str'>":
                _str += _msg_str
                _prev_int = False
            else:
                _str += (',' if _prev_int else ' ') + str(_msg_str)

        _file = os.path.basename(_file_name)

        print("[<%s> %s : %d]" %(_file, _func_name, _line_no), _str, file = sys.stderr)
    else:
        pass

