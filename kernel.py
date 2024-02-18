# Add this at the top of your file
import debugpy 

# Add this at the first spot you'd like to start debugging from
# 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
debugpy.listen(5678) # ensure that this port is the same as the one in your launch.json
print("Waiting for debugger attach")
debugpy.wait_for_client()
debugpy.breakpoint()
print('break on this line')
