dll_dir = "../dlls"
import os
os.environ['PATH'] = dll_dir + os.pathsep + os.environ['PATH']
import sys
sys.path.append(dll_dir)