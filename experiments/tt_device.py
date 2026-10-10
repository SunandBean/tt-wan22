"""Open / close the single P100a (mesh 1x1) and read DRAM use, for the scripts that need no fabric."""
import ttnn


def open_device(l1_small_size: int = 32768):
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=l1_small_size)


def close_device(dev):
    ttnn.close_mesh_device(dev)


def dram_stats(dev):
    ttnn.synchronize_device(dev)
    v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    banks = int(v.num_banks)
    return {"total_bytes": int(v.total_bytes_per_bank) * banks,
            "allocated_bytes": int(v.total_bytes_allocated_per_bank) * banks,
            "free_bytes": int(v.total_bytes_free_per_bank) * banks,
            "largest_free_bytes_per_bank": int(v.largest_contiguous_bytes_free_per_bank)}
