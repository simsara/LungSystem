# 用于远程调试
import ptvsd

# Allow other computers to attach to ptvsd at this IP address and port.
ptvsd.enable_attach(address=('192.168.1.112', 40068), redirect_output=True)

# Pause the program until a remote debugger is attached
ptvsd.wait_for_attach()

msg = "Hello World"
print(msg)
