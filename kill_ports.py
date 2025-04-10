#!/usr/bin/env python3
"""
Port Killer Script - Find and kill processes using specific ports
Usage: python kill_ports.py [port1] [port2] ...

Examples:
  python kill_ports.py 8080
  python kill_ports.py 8080 8081 8765
"""
import os
import sys
import subprocess
import platform

def get_process_on_port(port):
    """Get process information for a process using a specific port"""
    system = platform.system().lower()
    
    try:
        if system == 'darwin' or system == 'linux':  # macOS or Linux
            # Use lsof to find process
            cmd = f"lsof -i :{port} -t"
            result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
            pids = result.decode().strip().split('\n')
            return [pid for pid in pids if pid]
        
        elif system == 'windows':
            # Use netstat on Windows
            cmd = f"netstat -ano | findstr :{port}"
            result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
            lines = result.decode().strip().split('\n')
            pids = []
            for line in lines:
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.strip().split()
                    pids.append(parts[-1])
            return list(set(pids))  # Remove duplicates
        
        else:
            print(f"Unsupported operating system: {system}")
            return []
    
    except subprocess.CalledProcessError:
        # No process found using this port
        return []
    except Exception as e:
        print(f"Error getting process on port {port}: {e}")
        return []

def kill_process(pid):
    """Kill a process by PID"""
    system = platform.system().lower()
    
    try:
        if system == 'darwin' or system == 'linux':  # macOS or Linux
            # Get process name first
            cmd = f"ps -p {pid} -o comm="
            process_name = subprocess.check_output(cmd, shell=True).decode().strip()
            
            # Kill the process
            subprocess.check_output(f"kill -9 {pid}", shell=True)
            return True, process_name
        
        elif system == 'windows':
            # Get process name first
            cmd = f"tasklist /fi \"PID eq {pid}\" /fo list | findstr \"Image\""
            result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
            process_name = result.decode().strip().split(":")[-1].strip()
            
            # Kill the process
            subprocess.check_output(f"taskkill /F /PID {pid}", shell=True)
            return True, process_name
        
        else:
            print(f"Unsupported operating system: {system}")
            return False, "Unknown"
    
    except subprocess.CalledProcessError as e:
        print(f"Error killing process {pid}: {e}")
        return False, "Unknown"
    except Exception as e:
        print(f"Error: {e}")
        return False, "Unknown"

def main():
    """Main function"""
    # Print header
    print("\n===== Port Killer =====")
    print(f"Running on: {platform.system()} {platform.release()}")
    print("=======================\n")
    
    # Get ports from command line arguments
    if len(sys.argv) > 1:
        ports = [int(p) for p in sys.argv[1:] if p.isdigit()]
    else:
        # Default ports to check
        ports = [8080, 8081, 8765]
        print(f"No ports specified, checking default ports: {ports}")
    
    # Check each port
    for port in ports:
        print(f"\nChecking port {port}...")
        pids = get_process_on_port(port)
        
        if not pids:
            print(f"✓ No process found using port {port}")
            continue
        
        print(f"! Found {len(pids)} process(es) using port {port}")
        
        for pid in pids:
            print(f"  - PID {pid}: ", end="")
            success, process_name = kill_process(pid)
            
            if success:
                print(f"✓ Killed process '{process_name}' (PID: {pid})")
            else:
                print(f"✗ Failed to kill process (PID: {pid})")
    
    print("\n===== Finished =====\n")

if __name__ == "__main__":
    main()