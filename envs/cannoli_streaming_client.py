import socket, time, os, signal
import subprocess, pathlib, json, pickle, hashlib
from .bench import Bench

BUFFER_SIZE = 1024
POISON = 0x36afb081
INIT_EVENT_COUNT = 4
SCHEMAS = ["heap"]

# Things to think about / do
# 1. Hook up NAV & Tracer to this file -> anything that changes in the loop
# 2. Sub in invariant checking (Aaron's code)
# 3. Complete lockbox schema, lockbox tracing
# 4. Complete CRC schema, tracing
# 5. Unit test message parsing
# 6. Define input schema, action space


# TODO: create a universal timeout

class CannoliStreamingClient:

    def __init__(self, exec_name):
        """
        initializes qemu and cannoli socket connections
        """
        self.recv_buf: bytes = b''
        self.events = list()
        self.memos = dict()
        self.expected_lockbox = None
        self.pid = os.getpid()
        print("creating bench process")
        self.bench = Bench()
        print("start with bench")
        self.bench.run("CromulenceExample1", "nav", None, True)
        print("done with bench")

        # create receive socket for Cannoli feedback
        # must do this first so that Cannoli can initialize init_pid()
        self.recv_path_cannoli = "/tmp/nav_" + str(self.pid)
        print(f"receive path {self.recv_path_cannoli}")
        try:
            os.unlink(self.recv_path_cannoli)
        except OSError:
            if os.path.exists(self.recv_path_cannoli):
                raise

        self.cannoli_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.cannoli_sock.bind(self.recv_path_cannoli)
        self.cannoli_sock.listen(1)
        print("[*] NAV: listening on " + self.recv_path_cannoli)
        self.conn, addr = self.cannoli_sock.accept()
        # print("[*] NAV: received connection on " + self.recv_path_cannoli)
        self.conn.settimeout(10)

        # flush all init (pre-main) events and return the last one,
        # representative of the "initial state"
        # self._flush()

    def _flush(self) -> str:
        for _ in range(INIT_EVENT_COUNT ):
            self.try_read()
            # print(f"Init {self.try_read()}")

    def try_write(self, choice: tuple) -> None:
        """
        invoke executable with input string
        action: input string
        """
        data = pickle.dumps(choice)
        print("NAV->Tracer: ", choice)
        n = self.conn.send(data + b'\n')
        # print(f"[*] NAV: sent {n} bytes: {action.encode()} to qemu process")
        # self.qemu.stdin.flush()

    # cannoli_client packet format
    # POISON value as a u32 integer (4 bytes) followed by the schema length
    # for every possible schema (this must be synchronized b/w cannoli and
    # the receiving client). All lengths are little endian.
    #
    # schema lengths will be null for any event type that was not generated
    # by the current event.
    #
    # The current version of cannoli only generates 1 schema, so there is
    # only 1 length value and one data blob following the header.
    #
    #       u32            u32             u32       ...       u32
    # .-------------.---------------.--------------.-...-.--------------.
    # |    POISON   |  Schema 1 len | Schema 2 len | ... | Schema n len | ...
    # `-------------`---------------`--------------`-...-`--------------`
    #
    #       Schema x len    Schema y len
    #     .---------------.---------------.-----...------.
    # ... | Schema x data | Schema y data |     ...      |
    #     `---------------`---------------`-----...------`

    # renaming "try_read" per API screenshot
    def try_read(self) -> str:  # TO DO: update to schema return
        # try: # TO DO: holding off on this try right now because it's difficult
        # to debug errors when handling with an except statement
        # receive bytes
        while True:
            try:
                b = self.conn.recv(BUFFER_SIZE)
                if b == b'' and self.recv_buf != b'': break
                self.recv_buf += b
                # break if didn't receive a full buffer
                # print("received data: ", b)
                if len(b) != BUFFER_SIZE: break
            except Exception as e:
                print("Exception receiving", e)
                break

        # print("Received: ", self.recv_buf)
        self.parse()
        print("events: ", self.events)
        for e in self.events:
            invariants = self.check(e)
            print("got invariant: ", invariants)
        # return data to nav here
        self.events = list()

    def check(self, e: dict) -> tuple:
        def dict_hash(dictionary) -> str:
            dhash = hashlib.md5()
            encoded = json.dumps(dictionary, sort_keys=True).encode()
            dhash.update(encoded)
            return dhash.hexdigest()

        # check memos
        h = dict_hash(e)
        if h in self.memos:
            # print("got memoized state: ", h)
            return self.memos[h]

        # calculate invariant and store in hash table
        # if first time seeing lockbox, store initial value
        if self.expected_lockbox is None: self.expected_lockbox = e["lockbox"]
        invariants = [False, False]
        if e["lockpin"] == 0: invariants[0] = True
        if e["lockbox"] != self.expected_lockbox: invariants[1] = True
        self.memos[h] = invariants
        # print(self.memos)
        return invariants

    # parse packets stored in internal receive buffer and store events
    def parse(self) -> None:
        while True:
            if len(self.recv_buf) < 4:
                self.reset_buffer()
                return

            # check packet integrity by comparing poison value
            poison = int.from_bytes(self.recv_buf[:4], "little")
            if POISON != poison:
                # packet corrupted, flush until next poison is found
                self.reset_buffer()
                continue
            self.recv_buf = self.recv_buf[4:]

            # get schema lengths. Indexes correspond to the schemas defined in
            # SCHEMAS
            schema_lengths = []
            if len(self.recv_buf) < len(SCHEMAS) * 4:
                self.reset_buffer()
                continue
            for i in range(len(SCHEMAS)):
                schema_lengths.append(int.from_bytes(self.recv_buf[:4], "little"))
                self.recv_buf = self.recv_buf[4:]

            schema_data = {}
            # use schema lengths to parse schema data (if available)
            for i in range(len(SCHEMAS)):
                if len(self.recv_buf) < schema_lengths[i]:
                    self.reset_buffer()
                    continue

                # full event available in buffer. Parse and store event
                data = self.recv_buf[:schema_lengths[i]]
                self.recv_buf = self.recv_buf[schema_lengths[i]:]
                try:
                    # TO DO: can there be more than one schema returned??
                    # schema_data[SCHEMAS[i]] = json.loads(data.decode())
                    j = json.loads(data.decode())
                    self.events.append(j)
                    print(j)
                except:
                    # failure to deserialize, scrap rest of buffer until next
                    # poison
                    self.reset_buffer()
                    continue

                # TO DO: if returning more than one schema, would accumulate in
                # dict or list in `try` clause above and return here
                # return ??

            # the following exceptions return none:
            # socket.timeout, int conversion failure, buffer out of space
            # except:
            # return None

    # finds next poison value to reset buffer. If dne, null buffer out
    def reset_buffer(self) -> None:
        idx = self.recv_buf.find(POISON.to_bytes(4, "little"))
        if (idx >= 0):
            self.recv_buf = self.recv_buffer[idx:]
        else:
            self.recv_buffer = b''

    def reset(self):
        """
        kills socket connections
        """
        if self.bench:
            print("killing bench")
            self.bench.kill()
        print("killing qemu")
        subprocess.run(["killall", "qemu-system-x86_64"])
        self.conn.close()
        self.cannoli_sock.close()
        # signal.signal(signal.SIGINT, self._handle_kill)

    def _handle_kill(self):
        if self.qemu and self.qemu.poll():
            self.qemu.terminate()


# if __name__ == "__main__":
#     client = CannoliStreamingClient("blah")
#
#     # write to lockpin
#     print("\n[-] Write the value 1337 to the lockpin")
#     client.try_write((0, 1337))
#     recvd = client.try_read()
#     print("\n[-] Get the crc of 4 bytes at 0x677a300 (the lockpin address)")
#     client.try_write((2, 0x67f8380))
#     recvd = client.try_read()
#     print("\n[-] Write a 4-byte int (0) to the lockpin address (0x67f8380)")
#     client.try_write((1, 0x67f8380, 0))
#     recvd = client.try_read()
#     print("\n[-] Write a 4-byte int (12345678) to inside the lockbox (0x67f9100)")
#     client.try_write((1, 0x6a02100, 12345678))
#     recvd = client.try_read()

    # Joe's lockpin addr
    # client.try_write((1, 0x677a300, 0))

'''
    recv_and_print_pkt(client)

    print("\n[-] Write the value 7331 to the lockpin")
    client.try_write((0, 7331))
    recv_and_print_pkt(client)

    # call WriteData protocol function to write 1729 at 0x100 bytes into the lockbox
    # should fail to trigger lockbox buffer watchpoint as the lockpin is non-zero
    print("\n[-] Write a 4-byte int (1729) to inside the lockbox (0x67f9100)")
    client.try_write((1, 0x67f9100, 1729))
    recv_and_print_pkt(client)

    # Get the lockpin crc
    print("\n[-] Get the crc of 4 bytes at 0x677a300 (the lockpin address)")
    client.try_write((2, 0x677a300))
    recv_and_print_pkt(client)

    # Write 0 to the lockpin location using WriteData protocol
    print("\n[-] Write a 4-byte int (0) to the lockpin address (0x677a300)")
    client.try_write((1, 0x677a300, 0))
    recv_and_print_pkt(client)

    # try writing to the lockbox again, this time should succeed
    print("\n[-] Write a 4-byte int (123456789) to inside the lockbox (0x67f9100)")
    client.try_write((1, 0x67f9100, 12345678))
    recv_and_print_pkt(client)
'''
