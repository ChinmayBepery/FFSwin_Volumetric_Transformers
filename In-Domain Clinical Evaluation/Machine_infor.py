import platform
import psutil
import subprocess

# -------------------------------
# SYSTEM INFORMATION
# -------------------------------
def get_system_info():
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

# -------------------------------
# CPU INFORMATION
# -------------------------------
def get_cpu_info():
    info = {}
    try:
        import cpuinfo
        cpu = cpuinfo.get_cpu_info()
        info["brand"] = cpu.get("brand_raw", "N/A")
        info["architecture"] = cpu.get("arch", "N/A")
    except Exception:
        info["brand"] = platform.processor()
        info["architecture"] = platform.machine()

    info["cores_physical"] = psutil.cpu_count(logical=False)
    info["cores_logical"] = psutil.cpu_count(logical=True)
    freq = psutil.cpu_freq()
    info["max_frequency_mhz"] = f"{freq.max:.2f} MHz" if freq else "N/A"
    return info

# -------------------------------
# RAM INFORMATION
# -------------------------------
def get_ram_info():
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024 ** 3), 2),
        "available_gb": round(mem.available / (1024 ** 3), 2),
        "used_gb": round(mem.used / (1024 ** 3), 2),
        "usage_percent": mem.percent,
    }

# -------------------------------
# GPU INFORMATION (VM SAFE)
# -------------------------------
def get_gpu_info():
    gpus = []

    # 1️⃣ Try GPUtil
    try:
        import GPUtil
        for gpu in GPUtil.getGPUs():
            gpus.append({
                "name": gpu.name,
                "memory_total_mb": gpu.memoryTotal,
                "memory_used_mb": gpu.memoryUsed,
                "load_percent": round(gpu.load * 100, 2)
            })
        if gpus:
            return gpus
    except Exception:
        pass

    # 2️⃣ Try PyTorch (common in research VMs)
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_gb": round(props.total_memory / (1024 ** 3), 2),
                    "cuda_capability": f"{props.major}.{props.minor}"
                })
            return gpus
    except Exception:
        pass

    # 3️⃣ Fallback: nvidia-smi (most reliable in VMs)
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            encoding="utf-8"
        )
        for line in result.strip().split("\n"):
            name, mem = line.split(",")
            gpus.append({
                "name": name.strip(),
                "memory_total": mem.strip()
            })
        return gpus
    except Exception:
        pass

    return [{"error": "GPU not visible inside VM (driver / passthrough issue)"}]

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    print("\nSYSTEM INFO:", get_system_info())
    print("\nCPU INFO:", get_cpu_info())
    print("\nRAM INFO:", get_ram_info())
    print("\nGPU INFO:", get_gpu_info())
