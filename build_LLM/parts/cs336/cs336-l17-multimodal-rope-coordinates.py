import torch

def multimodal_rope_coordinates(segments: list) -> dict:
    """
    Returns a dict with coordinates, a CPU int64 tensor of shape (3, packed_length).
    """
    offset = 0
    chunks = []

    for seg in segments:
        seg_type = seg["type"]

        if seg_type == "text":
            length = seg["length"]
            if length == 0:
                continue
            pos = torch.arange(offset, offset + length, dtype=torch.int64)
            chunks.append(pos.unsqueeze(0).expand(3, -1))
            offset += length

        elif seg_type == "separator":
            chunks.append(torch.full((3, 1), offset, dtype=torch.int64))
            offset += 1

        elif seg_type == "image":
            H, W = seg["height"], seg["width"]
            if H == 0 or W == 0:
                continue
            rows = torch.arange(offset, offset + H, dtype=torch.int64)
            cols = torch.arange(offset, offset + W, dtype=torch.int64)
            grid_h, grid_w = torch.meshgrid(rows, cols, indexing="ij")
            flat_h = grid_h.reshape(-1)
            flat_w = grid_w.reshape(-1)
            flat_t = torch.full_like(flat_h, offset)
            chunks.append(torch.stack([flat_t, flat_h, flat_w], dim=0))
            offset += max(H, W)

        elif seg_type == "video":
            F, H, W = seg["frames"], seg["height"], seg["width"]
            if F == 0 or H == 0 or W == 0:
                continue
            stride = seg.get("frame_stride", 1)
            times = offset + torch.arange(F, dtype=torch.int64) * stride
            rows = torch.arange(offset, offset + H, dtype=torch.int64)
            cols = torch.arange(offset, offset + W, dtype=torch.int64)
            grid_t, grid_h, grid_w = torch.meshgrid(times, rows, cols, indexing="ij")
            chunks.append(
                torch.stack(
                    [grid_t.reshape(-1), grid_h.reshape(-1), grid_w.reshape(-1)],
                    dim=0,
                )
            )
            offset += max((F - 1) * stride + 1, H, W)

    if not chunks:
        coordinates = torch.empty((3, 0), dtype=torch.int64)
    else:
        coordinates = torch.cat(chunks, dim=1)

    return {"coordinates": coordinates}
