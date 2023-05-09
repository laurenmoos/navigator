#!/usr/bin/env python3

"""
bench.py

Python script for building and running edk2 with vulnerable modules installed
"""

import os
import yaml
import argparse
import subprocess
import signal
from pathlib import Path

class Bench:
    def __init__(self, debug=False):
        self.gdb = None
        self.debug = debug
        pass

    def clean(self):
        print("Cleaning EDK2 build...")
        edk2_dir = str(pkg_dir / "edk2")
        shell_source(pkg_dir / "edk2/edksetup.sh", edk2_dir)
        subprocess.run(["build", "clean"], cwd=edk2_dir)
        clean_symlinks()

    def build(self, module: str) -> int:
        print(f"[-] Building edk2 with vuln module '{module}'")
        edk2_dir = str(pkg_dir / "edk2")
        subprocess.run(["make", "-C", "BaseTools"], cwd=edk2_dir)
        shell_source(pkg_dir / "edk2/edksetup.sh", edk2_dir)

        # TODO pull new edk2 here

        print(f"[-] Checking out module \"{module}\"")
        out = subprocess.run(["git", "checkout", module], cwd=edk2_dir)
        if out.returncode != 0:
            print("[!] Dirty tree conflict. edk2 directory cannot " + \
                  "checkout vulnerable module branch")
            resp = input("    git [c]lean or git [s]tash or [Q]uit? ")
            if resp == "c":
                subprocess.run(["git", "status", "-s", "-unormal"],
                    cwd=edk2_dir)
                confirm = input("Really delete the above files? y/N ")
                if confirm == "y":
                    subprocess.run(["git", "clean", "-df"], cwd=edk2_dir)
                else:
                    return 1
            if resp == "s":
                subprocess.run(["git", "stash"], cwd=edk2_dir)
            elif resp == "q":
                return 0
            else:
                exit(1)

            # checkout again
            subprocess.run(["git", "checkout", module], cwd=edk2_dir)

        # actually build and create the approapriate symlinks
        subprocess.run(["build"], shell=True, cwd=edk2_dir)
        create_symlinks(module)
        return 0

    def run(self, module: str, gdb_script: str, efi_vars: str = None,
            vertical: bool = None) -> int:
        # Check that the module we are running is the one we built
        # if not, offer to build it first
        out = subprocess.run(["git", "branch", "--show-current"],
                             cwd=(pkg_dir / "edk2"), capture_output=True)
        curr_mod = out.stdout.decode().strip()
        if curr_mod != module:
            print(f"[!] Running different edk2 module than currently built: " +\
                   "\"{curr_mod}\"")
            resp = input(f"    [b]uild module {module} or [q]uit? ")
            if resp == "b":
                rc = self.build(module)
                if rc != 0: return rc
                # out = subprocess.run(["python3", (pkg_dir / "bench.py"),
                #                       "build", module])
                # if out.returncode != 0:
                #     exit(out.returncode)
            else:
                return int(resp == "q")

        # check for gdb script
        gdb_script_loc = pkg_dir / f"zoo/{module}/{gdb_script}.py"
        if not gdb_script_loc.exists():
            print(f"Cannot find gdb script at: {gdb_script_loc})")
            print(f"Available gdb scripts in {gdb_script_loc.parent} include:")
            gdb_scripts = filter(lambda x: x.suffix == ".py",
                                 gdb_script_loc.parent.iterdir())
            print("\n".join(map(lambda x: str(x.stem), gdb_scripts)))
            return 1

        # set initial uefi variable state
        if efi_vars:
            if custom_vars_path.exists():
                print(f"[-] Custom UEFI Variable firmware already exists: " + \
                       "{custom_vars_path}")
                print("    deleting...")
                # TODO prompt for overwrite? defaults to delete
                custom_vars_path.unlink()

            defined_vars_path = pkg_dir / f"zoo/{module}/{efi_vars}.vars"
            if not defined_vars_path.exists():
                print(f"[!] Cannot find custom defined UEFI variables at " + \
                       "{defined_vars_path}")
            return 1

            print("[-] Compiling custom UEFI variable firmware volume")
            subprocess.run(["ovmfvartool", "compile", str(defined_vars_path),
                           str(custom_vars_path)])
        else:
            print("[-] No custom variable file specified, using blank UEFI " + \
                  "Variable FV")
            subprocess.run(["ovmfvartool", "generate-blank",
                           str(custom_vars_path)])

        # build qemu command and run
        print(f"Running OVMF Module {module}")
        qemu_cmd = ""
        qemu_cmd += f"qemu-system-x86_64 -L {pkg_dir / 'enclosure'} "
        qemu_cmd += f"-drive if=pflash,format=raw,unit=0,file={code_path} "
        qemu_cmd += f"-drive if=pflash,format=raw,unit=1,file={custom_vars_path} "
        qemu_cmd += "-machine q35 "
        qemu_cmd += "-boot menu=on,splash-time=3 "
        qemu_cmd += "-net none -S -s "
        qemu_cmd += f"-hda fat:rw:{pkg_dir / 'enclosure/hd/'} "
        qemu_cmd += "-debugcon file:debug.log -global isa-debugcon.iobase=0x402 "
        qemu_cmd += "-serial mon:stdio -nographic "
        print(f"Using QEMU command: {qemu_cmd}")

        subprocess.run(["tmux",
                        "splitw",
                        "-f" if vertical else "-fh",
                        qemu_cmd])

        # adds the current directory to pythonpath so gdb scripts don't need to
        curpath = os.environ.get('PYTHONPATH')
        if curpath is not None:
            os.environ['PYTHONPATH'] = str(pkg_dir) + os.pathsep + curpath
        else:
            os.environ['PYTHONPATH'] = str(pkg_dir)

        # run gdb with appropriate script
        # currently requires a self-compiled gdb as the default ubuntu one doesn't
        # have the Windows osabi
        gdb_dir = pkg_dir / "binutils-gdb/gdb"
        gdb_cmd = [str(gdb_dir / "gdb"), "--data-directory",
                   str(gdb_dir / "data-directory"),
                   "-q", "--nx", "-x", str(gdb_script_loc)]
        if self.debug:
            subprocess.run(gdb_cmd)
        else:
            self.gdb = subprocess.Popen(gdb_cmd)

    def kill(self):
        if self.gdb and self.gdb.poll() is None:
            print("killing gdb")
            self.gdb.terminate()

# store root directory
pkg_dir = Path(os.environ["UEFI_PATH"])

# get a list of module names and set OVMF code and vars location '
modules = list(map(lambda x: x.name, (pkg_dir / "zoo").iterdir()))
code_path = pkg_dir / "edk2/Build/OvmfX64/DEBUG_GCC5/FV/OVMF_CODE.fd"
vars_path = pkg_dir / "edk2/Build/OvmfX64/DEBUG_GCC5/FV/OVMF_VARS.fd"
custom_vars_path = pkg_dir / "edk2/Build/OvmfX64/DEBUG_GCC5/FV/OVMF_VARS_CUSTOM.fd"


# set up argument parser and syntatic sugar to make subcommands a python function
# taken from:
# https://mike.depalatis.net/blog/simplifying-argparse.html
cli = argparse.ArgumentParser(description="List of vulnerable modules: " + " ".join(modules))
subparsers = cli.add_subparsers(dest="subcommand")
def subcommand(args=[], parent=subparsers):
    def decorator(func):
        parser = parent.add_parser(func.__name__, description=func.__doc__)
        for arg in args:
            parser.add_argument(*arg[0], **arg[1])
        parser.set_defaults(func=func)
    return decorator


def argument(*name_or_flags, **kwargs):
    return ([*name_or_flags], kwargs)

# loads the config file dictionary for a given module
def load_config(module: str):
    config_path = pkg_dir / f"zoo/{module}/config.yaml"
    if not config_path.exists():
        print(f"Cannot find config at {config_path}")
        exit(1)
    config = yaml.load(config_path.read_text(),
                       Loader=yaml.Loader)
    return config


# Runs a script from a directory (cwd) and updates the current process
# with any environment variable changes that occurred
# A bit hacky but necessary for running the edk2setup.sh script. Taken from:
# https://stackoverflow.com/questions/7040592/calling-the-source-command-from-subprocess-popen
def shell_source(script, cwd):
    print(f"Sourcing {script} in directory {cwd}")
    out = subprocess.run(f"source {script} > /dev/null; cat /proc/$$/environ",
                         stdout=subprocess.PIPE,
                         shell=True,
                         executable="/bin/bash",
                         cwd=cwd).stdout.strip(b"\x00").decode()
    env = dict((evar.split("=", 1) for evar in out.split("\x00")))
    os.environ.update(env)


@subcommand([argument("module", choices=modules, help="Vulnerable module to build"),
             ])
def build(args):
    "Build edk given a specific module, resolving any dirty tree git conflicts"
    bench = Bench()
    return_code = bench.build(args.module)
    print(f"Finished building with return code {return_code}")

@subcommand()
def clean(args):
    "Cleans build directory using edk2's clean command"
    bench = Bench()
    bench.clean()

def clean_symlinks():
    "Remove symlinks to a specific vulnerable module from the QEMU runtime dir"
    for p in (pkg_dir / "enclosure/hd").iterdir():
        if not p.name.startswith("."):
            print(f"unlinking {p}")
            p.unlink()

def create_symlinks(module):
    "Create symlinks to a specific vulnerable module for the QEMU runtime dir"
    clean_symlinks()
    config = load_config(module)
    for fs_symlink in config['fs_symlinks']:
        link = pkg_dir / "enclosure/hd" / Path(fs_symlink).name
        target = (pkg_dir / fs_symlink).absolute()
        print(f"linking {link} to {target}")
        link.symlink_to(target)


# Run the vulnerable module in QEMU with the specified gdb script
@subcommand([argument("module", choices=modules, help="Vulnerable module to build"),
             argument("-v", "--vertical", action="store_true", help="Split tmux pane vertically instead of horizontally"),
             argument("-g", "--gdb-script", default="debug", help="Which gdb script to run (stored in zoo/<module>/<script>.py"),
             argument("-e", "--efi-vars", help="Which efi variable file to install (stored in zoo/<module>/<yaml-file>.vars")
              ])

def run(args):
    "Run the vulnerable module in QEMU with the specified gdb script"
    bench = Bench(debug=True)
    efi_vars = args.efi_vars if args.efi_vars else None
    vertical = args.vertical if args.vertical else None
    bench.run(args.module, args.gdb_script, efi_vars, vertical)

# Lists the vulnerable modules
@subcommand()
def list(args):
    for module in modules:
        print(module)


@subcommand([argument("module", choices=modules,
                      help="Vulnerable module to print help text for")])
def help(args):
    " Prints the help text for a specific vulnerable module"
    # TODO: specify a locaiton for this per module
    print("Get help text for vuln modules")


# def kill(signal, frame):
#     global gdb_proc
#     print("killing gdb")
#     if gdb_proc is not None and gdb_proc.poll() is None: gdb_proc.terminate()
#     exit(1)
#
# signal.signal(signal.SIGTERM, kill)
# signal.signal(signal.SIGINT,  kill)
# signal.signal(signal.SIGQUIT, kill)

if __name__ == "__main__":
    args = cli.parse_args()
    if args.subcommand is None:
        cli.print_help()
    else:
        args.func(args)
