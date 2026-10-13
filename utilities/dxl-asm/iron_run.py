"""Run a command in iron_env.ps1's environment (ironenv python, Peano, XRT) at BelowNormal; python scripts use ironenv's python."""
import ctypes, os, subprocess, sys

ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
SP = r"C:\dev\mlir-aie\ironenv\Lib\site-packages"
env = dict(os.environ)
env["VIRTUAL_ENV"] = r"C:\dev\mlir-aie\ironenv"
env["MLIR_AIE_INSTALL_DIR"] = SP + r"\mlir_aie"
env["PEANO_INSTALL_DIR"] = SP + r"\llvm-aie"
env.pop("XILINX_XRT", None)
env["XRT_ROOT"] = r"C:\Xilinx\XRT"
env["PYTHONPATH"] = r"C:\Xilinx\XRT\python" + (";" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
env["NPU2"] = "1"
env["PATH"] = ";".join([r"C:\Windows\System32\AMD", r"C:\Xilinx\XRT", r"C:\Xilinx\XRT\lib", SP + r"\mlir_aie\lib",
                        SP + r"\mlir_aie\bin", r"C:\dev\mlir-aie\ironenv\Scripts", r"C:\Program Files\LLVM\bin", env["PATH"]])
cmd = sys.argv[1:]
if cmd[0] == "python":
    cmd[0] = r"C:\dev\mlir-aie\ironenv\Scripts\python.exe"
sys.exit(subprocess.run(cmd, env=env).returncode)
