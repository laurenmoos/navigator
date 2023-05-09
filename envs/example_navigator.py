import socket, time, os, signal
import subprocess, pathlib 

DEBUG = False
qemu = None
conn = None 

binaries = "/Users/lauren/leetasm-private/cannoli-mvp/binaries"
data = "/Users/lauren/leetasm-private/cannoli-mvp/data"

def handle_kill(signum, frame):
    global qemu
    if qemu != None and qemu.poll() == None:
        qemu.terminate()

def read_from_cannoli():
    global conn
    f = open("./cannoli_output", "w")
    while True:
       try:
           f.write(conn.recv(1024).decode())
       except TimeoutError:
           print("Timing out")
           break

    f.close()
    
def try_read(): 
    global qemu
    try:
        res, err = qemu.stdout.read()
    except subprocess.TimeoutExpired:
        print("[*] NAV: timeout trying to read from subprocess")
    except Exception as e:
        print("[!] NAV: uncaught exception reading from subprocess")
        print(e)
        exit(1)

def main():
    global qemu, conn, data, binaries
    # read input data 
    f = open(str(data) + "/input_32.bin", "rb")
    data = f.read()
    print(f"[*] NAV: generated {len(data)} bytes of input")    
    # exec cannoli and get QEMU child process PID 
    if not DEBUG:
        print("[*] NAV: spawning qemu process")
        qemu = subprocess.Popen([
                      "/home/harden/Documents/harden/qemu/build/qemu-mipsel", 
                      "-cannoli", 
                      "/home/harden/Documents/harden/mvp/target/release/libmvp.so", 
                      str(binaries) + "/wrap_and_win_mips"],
                      stdin=subprocess.PIPE,
                      stdout=subprocess.PIPE)
    else: 
        print("[*] NAV: spawning debugger QEMU process")
        qemu = subprocess.Popen([
                      "/home/harden/Documents/harden/qemu/build/qemu-mipsel", 
                      "-g", "9999",
                      str(binaries) + "./wrap_and_win_mips"])
   
    # create receive socket for Cannoli feedback
    # must do this first so that Cannoli can initialize init_pid()
    pid = os.getpid()
    recv_path = "/tmp/nav_" + str(pid) 
    try:
        os.unlink(recv_path)
    except OSError:
        if os.path.exists(recv_path):
            raise
    
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(recv_path)
    s.listen(1)
    print("[*] NAV: listening on " + recv_path)
    conn, addr = s.accept()
    print("[*] NAV: received connection on " + recv_path)
    conn.settimeout(5)
    
    # spawn child process which listens for cannoli feedback
    p = os.fork()

    if p == 0:
        read_from_cannoli()

    else: 
       
        print("[*] NAV: sending data to qemu process")
        # send data
        try: 
            n = qemu.stdin.write(data)
            qemu.stdin.flush()
            print(f"[*] NAV: sent {n} bytes to qemu process")
        except subprocess.TimeoutExpired:
            print("[!] NAV: timeout expired trying to write data to qemu process")
            if qemu.poll() != None: 
                print("   [!] NAV: qemu process died")
                exit(1)
            # get stdout data from qemu process
            res, err = qemu.communicate(timeout=5)
            print(res)
            qemu.terminate()
            exit(1)
        except Exception as e:
            print(e)
            exit(1)
    
        print(f"[*] NAV: waiting for child process to finish")
        os.waitpid(p, 0)
    
        print(f"[*] NAV: child process exited {p}")
        # cleanup process 
        print(f"[*] NAV: killing qemu process {qemu.pid}")
        qemu.terminate()
        print(f"[*] NAV: finishing python3 process {pid}")

if __name__ == "__main__":
    # handle Ctrl+C to kill QEMU in case it does not terminate naturally
    signal.signal(signal.SIGINT, handle_kill)

    main()
