"""RAUNE-Net OKLab chroma transfer — single-process, 10-bit.

Decodes input video via PyAV (rgb48le for 10-bit preservation), processes
on GPU with int32 frame cache to avoid redundant PCIe transfers, encodes
output via PyAV (rgb48le → yuv422p10le for ProRes).

Pipeline per batch:
  1. PyAV decode (rgb48le uint16) → numpy → int32 GPU upload (cached)
  2. Downscale to proxy on GPU (torch.interpolate)
  3. Batch RAUNE inference on proxy (fp16)
  4. OKLab delta computation at proxy on GPU (fp32)
  5. Upscale deltas to full-res on GPU
  6. Full-res OKLab transfer via fused Triton kernel (fp32 in, fp16 out)
  7. GPU download (uint16) → PyAV encode (rgb48le)
"""

import sys
import argparse
import time
import threading
import queue
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# Try to import Triton for fused kernel; fall back to PyTorch OKLab if unavailable
_USE_TRITON = False
try:
    import triton
    import triton.language as tl
    _USE_TRITON = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# OKLab conversion (matches dorea-color/src/lab.rs)
# Rescaled to CIELab-compatible ranges: L×100, a×300, b×300
# ═══════════════════════════════════════════════════════════════════════════════

_L_SCALE = 100.0
_AB_SCALE = 300.0

def srgb_to_linear(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(c):
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c.clamp(min=1e-7) ** (1.0 / 2.4) - 0.055)

def rgb_to_lab(rgb):
    """(N,3,H,W) sRGB [0,1] → (N,3,H,W) OKLab (rescaled to CIELab ranges)."""
    lin = srgb_to_linear(rgb)
    r, g, b = lin[:,0:1], lin[:,1:2], lin[:,2:3]
    l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
    m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
    s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
    l_ = l.clamp(min=0.0).pow(1.0/3.0)
    m_ = m.clamp(min=0.0).pow(1.0/3.0)
    s_ = s.clamp(min=0.0).pow(1.0/3.0)
    L = (0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_) * _L_SCALE
    a = (1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_) * _AB_SCALE
    b_lab = (0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_) * _AB_SCALE
    return torch.cat([L, a, b_lab], dim=1)

def lab_to_rgb(lab):
    """(N,3,H,W) OKLab (rescaled) → (N,3,H,W) sRGB [0,1]."""
    L, a, b_lab = lab[:,0:1], lab[:,1:2], lab[:,2:3]
    ok_l = L / _L_SCALE
    ok_a = a / _AB_SCALE
    ok_b = b_lab / _AB_SCALE
    l_ = ok_l + 0.3963377774*ok_a + 0.2158037573*ok_b
    m_ = ok_l - 0.1055613458*ok_a - 0.0638541728*ok_b
    s_ = ok_l - 0.0894841775*ok_a - 1.2914855480*ok_b
    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_
    r =  4.0767416613*l - 3.3077115904*m + 0.2309699287*s
    g = -1.2684380041*l + 2.6097574007*m - 0.3413193963*s
    b =  -0.0041960863*l - 0.7034186145*m + 1.7076147010*s
    return linear_to_srgb(torch.cat([r, g, b], dim=1).clamp(min=0.0)).clamp(0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Fused Triton kernel — entire OKLab transfer in one kernel launch
# ═══════════════════════════════════════════════════════════════════════════════

if _USE_TRITON:
    @triton.jit
    def _oklab_transfer_kernel(
        frame_ptr, delta_ptr, out_ptr,
        n_pixels,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One kernel: sRGB→linear→OKLab→+delta→OKLab⁻¹→linear→sRGB per pixel."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_pixels

        sr = tl.load(frame_ptr + 0 * n_pixels + offs, mask=mask).to(tl.float32)
        sg = tl.load(frame_ptr + 1 * n_pixels + offs, mask=mask).to(tl.float32)
        sb = tl.load(frame_ptr + 2 * n_pixels + offs, mask=mask).to(tl.float32)

        threshold = 0.04045
        lr = tl.where(sr <= threshold, sr / 12.92,
                      tl.extra.cuda.libdevice.pow(((sr + 0.055) / 1.055), 2.4))
        lg = tl.where(sg <= threshold, sg / 12.92,
                      tl.extra.cuda.libdevice.pow(((sg + 0.055) / 1.055), 2.4))
        lb = tl.where(sb <= threshold, sb / 12.92,
                      tl.extra.cuda.libdevice.pow(((sb + 0.055) / 1.055), 2.4))

        l = 0.4122214708*lr + 0.5363325363*lg + 0.0514459929*lb
        m = 0.2119034982*lr + 0.6806995451*lg + 0.1073969566*lb
        s = 0.0883024619*lr + 0.2817188376*lg + 0.6299787005*lb

        l = tl.maximum(l, 0.0)
        m = tl.maximum(m, 0.0)
        s = tl.maximum(s, 0.0)
        l_ = tl.extra.cuda.libdevice.cbrt(l)
        m_ = tl.extra.cuda.libdevice.cbrt(m)
        s_ = tl.extra.cuda.libdevice.cbrt(s)

        ok_L = 0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_
        ok_a = 1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_
        ok_b = 0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_

        dL = tl.load(delta_ptr + 0 * n_pixels + offs, mask=mask).to(tl.float32)
        da = tl.load(delta_ptr + 1 * n_pixels + offs, mask=mask).to(tl.float32)
        db = tl.load(delta_ptr + 2 * n_pixels + offs, mask=mask).to(tl.float32)

        # Delta is in rescaled OKLab — unscale before adding
        ok_L = ok_L + dL / 100.0
        ok_a = ok_a + da / 300.0
        ok_b = ok_b + db / 300.0

        l2_ = ok_L + 0.3963377774*ok_a + 0.2158037573*ok_b
        m2_ = ok_L - 0.1055613458*ok_a - 0.0638541728*ok_b
        s2_ = ok_L - 0.0894841775*ok_a - 1.2914855480*ok_b

        l2 = l2_ * l2_ * l2_
        m2 = m2_ * m2_ * m2_
        s2 = s2_ * s2_ * s2_

        or_ =  4.0767416613*l2 - 3.3077115904*m2 + 0.2309699287*s2
        og  = -1.2684380041*l2 + 2.6097574007*m2 - 0.3413193965*s2
        ob  = -0.0041960863*l2 - 0.7034186145*m2 + 1.7076147010*s2

        or_ = tl.maximum(or_, 0.0)
        og  = tl.maximum(og, 0.0)
        ob  = tl.maximum(ob, 0.0)

        threshold2 = 0.0031308
        fr = tl.where(or_ <= threshold2, or_ * 12.92,
                      1.055 * tl.extra.cuda.libdevice.pow(or_, 1.0/2.4) - 0.055)
        fg = tl.where(og <= threshold2, og * 12.92,
                      1.055 * tl.extra.cuda.libdevice.pow(og, 1.0/2.4) - 0.055)
        fb = tl.where(ob <= threshold2, ob * 12.92,
                      1.055 * tl.extra.cuda.libdevice.pow(ob, 1.0/2.4) - 0.055)

        fr = tl.minimum(tl.maximum(fr, 0.0), 1.0)
        fg = tl.minimum(tl.maximum(fg, 0.0), 1.0)
        fb = tl.minimum(tl.maximum(fb, 0.0), 1.0)

        tl.store(out_ptr + 0 * n_pixels + offs, fr.to(tl.float16), mask=mask)
        tl.store(out_ptr + 1 * n_pixels + offs, fg.to(tl.float16), mask=mask)
        tl.store(out_ptr + 2 * n_pixels + offs, fb.to(tl.float16), mask=mask)


def triton_oklab_transfer(frame_nchw_f32, delta_nchw_f32):
    """Fused Triton kernel: entire OKLab transfer in one kernel launch."""
    _, C, H, W = frame_nchw_f32.shape
    # Pass fp32 to kernel — avoid precision loss before nonlinear transforms.
    # Kernel output stays fp16 (sufficient for 10-bit).
    frame_flat = frame_nchw_f32.squeeze(0).reshape(3, -1).contiguous()   # fp32
    delta_flat = delta_nchw_f32.squeeze(0).reshape(3, -1).contiguous()   # fp32
    n_pixels = H * W
    out_flat = torch.empty(3, n_pixels, dtype=torch.float16, device=frame_flat.device)

    BLOCK_SIZE = 1024
    grid = ((n_pixels + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _oklab_transfer_kernel[grid](
        frame_flat, delta_flat, out_flat,
        n_pixels=n_pixels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_flat.reshape(1, 3, H, W).float()


def pytorch_oklab_transfer(frame_nchw_f32, delta_nchw_f32):
    """PyTorch fallback: OKLab transfer without Triton."""
    lab = rgb_to_lab(frame_nchw_f32)
    lab = lab + delta_nchw_f32
    return lab_to_rgb(lab)


# ═══════════════════════════════════════════════════════════════════════════════
# Single-process mode (PyAV decode + encode, zero pipes)
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_process(args, model, normalize, model_dtype, _use_trt=False):
    """Decode → GPU process → encode, all in one process via PyAV."""
    import av

    fw, fh = args.full_width, args.full_height
    pw, ph = args.proxy_width, args.proxy_height
    batch_size = args.batch_size

    # Select transfer function
    if _USE_TRITON:
        transfer_fn = triton_oklab_transfer
        # Warm up Triton compilation
        _dummy_f = torch.zeros(1, 3, 16, 16, device="cuda")
        _dummy_d = torch.zeros(1, 3, 16, 16, device="cuda")
        transfer_fn(_dummy_f, _dummy_d)
        torch.cuda.synchronize()
        print("[raune-filter] Using Triton fused kernel", file=sys.stderr, flush=True)
    else:
        transfer_fn = pytorch_oklab_transfer
        print("[raune-filter] Using PyTorch OKLab (Triton unavailable)", file=sys.stderr, flush=True)

    # Open input
    in_container = av.open(args.input)
    in_stream = in_container.streams.video[0]
    in_stream.thread_type = "AUTO"
    total_frames = in_stream.frames or 0
    from fractions import Fraction
    fps_frac = in_stream.average_rate or in_stream.rate or Fraction(30, 1)
    fps = float(fps_frac)

    # Open output
    out_container = av.open(args.output, mode="w")
    codec_name = args.output_codec or "prores_ks"
    out_stream = out_container.add_stream(codec_name, rate=fps_frac)
    out_stream.width = fw
    out_stream.height = fh
    out_stream.thread_type = "AUTO"

    # Configure codec
    if codec_name == "prores_ks":
        out_stream.pix_fmt = "yuv422p10le"
        out_stream.options = {"profile": "3"}  # ProRes 422 HQ
    elif codec_name in ("hevc", "libx265"):
        out_stream.pix_fmt = "yuv420p10le"
    else:
        out_stream.pix_fmt = "yuv420p"

    print(f"[raune-filter] single-process: {fw}x{fh}, proxy={pw}x{ph}, "
          f"batch={batch_size}, codec={codec_name}, fps={fps:.3f}, "
          f"dtype={model_dtype}",
          file=sys.stderr, flush=True)

    # ─── 3-thread pipeline: decoder → GPU → encoder ────────────────────────
    # Bounded queues provide backpressure (memory bound: ~200MB at 4K)
    q_decoded = queue.Queue(maxsize=2)   # holds: list[np.ndarray] uint16 (~794 MB at 4K, batch=8)
    q_processed = queue.Queue(maxsize=2) # holds: list[np.ndarray] (one batch)

    # Shared error state
    errors: list[BaseException] = []
    errors_lock = threading.Lock()
    stop_event = threading.Event()

    def record_error(exc: BaseException) -> None:
        with errors_lock:
            errors.append(exc)
        stop_event.set()

    def put_or_stop(q: "queue.Queue", item, stop_event: threading.Event,
                    poll_interval: float = 0.1) -> bool:
        """Put item on queue, periodically checking stop_event.

        Returns True if put succeeded, False if stop_event was set first.
        Ensures worker threads cannot get stuck forever in a blocking put()
        when the downstream consumer has died.
        """
        while not stop_event.is_set():
            try:
                q.put(item, timeout=poll_interval)
                return True
            except queue.Full:
                continue
        return False

    # Frame counter shared with encoder thread for progress reporting
    t_start = time.time()
    encoded_count = 0

    # Per-stage cumulative timing (busy time, excluding queue waits)
    decode_busy = 0.0
    gpu_busy = 0.0
    encode_busy = 0.0

    # ─── Thread 1: Decoder ─────────────────────────────────────────────────
    def decoder_thread() -> None:
        nonlocal decode_busy
        try:
            batch: list[np.ndarray] = []
            for packet in in_container.demux(in_stream):
                if stop_event.is_set():
                    return
                # Time the entire packet.decode() iteration so that the
                # libavcodec decode work (which actually runs here) is
                # captured, not just the to_ndarray() conversion.
                t_decode_start = time.perf_counter()
                frames = list(packet.decode())
                decode_busy += time.perf_counter() - t_decode_start
                for frame in frames:
                    if stop_event.is_set():
                        return
                    t0 = time.perf_counter()
                    rgb = frame.to_ndarray(format="rgb48le")  # (H, W, 3) uint16
                    if rgb.shape[1] != fw or rgb.shape[0] != fh:
                        # PyAV reformat preserves 16-bit (PIL .to_image() is 8-bit)
                        frame_resized = frame.reformat(width=fw, height=fh, format="rgb48le")
                        rgb = frame_resized.to_ndarray(format="rgb48le")
                    decode_busy += time.perf_counter() - t0
                    batch.append(rgb)
                    if len(batch) >= batch_size:
                        if not put_or_stop(q_decoded, batch, stop_event):
                            return
                        batch = []
            if batch:
                if not put_or_stop(q_decoded, batch, stop_event):
                    return
        except BaseException as e:
            record_error(e)
        finally:
            # Sentinel for GPU thread; use put_or_stop so we cannot hang
            # here if the consumer has already died.
            put_or_stop(q_decoded, None, stop_event)

    # ─── Thread 2: GPU processing ──────────────────────────────────────────
    def gpu_thread() -> None:
        nonlocal gpu_busy
        try:
            while True:
                if stop_event.is_set():
                    return
                batch = q_decoded.get()
                if batch is None:
                    return
                t0 = time.perf_counter()
                try:
                    results = _process_batch(
                        batch, model, normalize,
                        fw, fh, pw, ph, transfer_fn,
                        model_dtype, _use_trt=_use_trt,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(
                        f"[raune-filter] CUDA out of memory at batch={batch_size}. "
                        f"Reduce with --direct-batch-size {max(1, batch_size // 2)}",
                        file=sys.stderr, flush=True,
                    )
                    raise
                gpu_busy += time.perf_counter() - t0
                if not put_or_stop(q_processed, results, stop_event):
                    return
        except BaseException as e:
            record_error(e)
        finally:
            # Sentinel for encoder thread; use put_or_stop so we cannot hang.
            put_or_stop(q_processed, None, stop_event)

    # ─── Thread 3: Encoder ─────────────────────────────────────────────────
    def encoder_thread() -> None:
        nonlocal encoded_count, encode_busy
        try:
            while True:
                if stop_event.is_set():
                    return
                results = q_processed.get()
                if results is None:
                    return
                for result_np in results:
                    t0 = time.perf_counter()
                    out_frame = av.VideoFrame.from_ndarray(result_np, format="rgb48le")
                    out_frame.pts = encoded_count
                    for pkt in out_stream.encode(out_frame):
                        out_container.mux(pkt)
                    encode_busy += time.perf_counter() - t0
                    encoded_count += 1
                    if encoded_count % (batch_size * 4) == 0:
                        elapsed = time.time() - t_start
                        fps_actual = encoded_count / elapsed if elapsed > 0 else 0
                        pct = (encoded_count / total_frames * 100
                               if total_frames else 0)
                        print(f"[raune-filter] {encoded_count} frames "
                              f"({pct:.0f}%, {fps_actual:.1f} fps)",
                              file=sys.stderr, flush=True)
        except BaseException as e:
            record_error(e)
            # On error, drain q_processed so the upstream gpu_thread does
            # not deadlock in put_or_stop on a full queue. record_error
            # has already set stop_event — once gpu_thread's in-flight put
            # lands in the slot we empty here, its next loop iteration
            # sees stop_event and exits cleanly.
            try:
                while True:
                    _ = q_processed.get_nowait()
            except queue.Empty:
                pass

    # Spawn threads
    t_dec = threading.Thread(target=decoder_thread, name="decoder", daemon=False)
    t_gpu = threading.Thread(target=gpu_thread, name="gpu", daemon=False)
    t_enc = threading.Thread(target=encoder_thread, name="encoder", daemon=False)

    t_dec.start()
    t_gpu.start()
    t_enc.start()

    # Wait for all threads to finish
    t_dec.join()
    t_gpu.join()
    t_enc.join()

    # Determine whether any worker thread raised; read under lock.
    error_to_raise: BaseException | None = None
    with errors_lock:
        if errors:
            error_to_raise = errors[0]

    # Flush encoder only on success — flushing after a failure can write
    # trailing packets to an already-broken output file.
    if error_to_raise is None:
        try:
            for pkt in out_stream.encode():
                out_container.mux(pkt)
        except BaseException as flush_err:
            error_to_raise = flush_err

    # Always close containers, but do not let a secondary close() failure
    # (e.g. disk-full on trailer write) mask the original error.
    try:
        out_container.close()
    except BaseException as close_err:
        if error_to_raise is None:
            error_to_raise = close_err
    try:
        in_container.close()
    except BaseException:
        pass  # input close errors are not interesting

    # If anything went wrong, delete the partial/corrupt output file so
    # callers do not see a broken ProRes left on disk.
    if error_to_raise is not None:
        try:
            import os
            if os.path.exists(args.output):
                os.remove(args.output)
                print(f"[raune-filter] removed partial output: {args.output}",
                      file=sys.stderr, flush=True)
        except OSError:
            pass
        raise error_to_raise

    elapsed = time.time() - t_start
    fps_actual = encoded_count / elapsed if elapsed > 0 else 0
    n = max(encoded_count, 1)
    print(f"[raune-filter] done: {encoded_count} frames in {elapsed:.1f}s "
          f"({fps_actual:.2f} fps)",
          file=sys.stderr, flush=True)
    print(f"[raune-filter] stage timing (busy ms/frame): "
          f"decode={decode_busy*1000/n:.1f} "
          f"gpu={gpu_busy*1000/n:.1f} "
          f"encode={encode_busy*1000/n:.1f} "
          f"wall={elapsed*1000/n:.1f}",
          file=sys.stderr, flush=True)
    return encoded_count


def _process_batch(batch_frames_np, model, normalize, fw, fh, pw, ph, transfer_fn, model_dtype, _use_trt=False):
    """Process a batch of frames on GPU. Returns list of uint16 HWC numpy arrays."""
    n = len(batch_frames_np)
    results = []

    with torch.no_grad():
        # Upload once as int32, keep on GPU for reuse in apply loop.
        # int32 because PyTorch has no uint16 — torch.from_numpy(uint16)
        # reinterprets as int16 (signed), corrupting values > 32767.
        full_gpu_cache = []
        proxy_tensors = []
        # Pre-allocate pinned upload buffer (reused per frame, pin once not per-frame)
        pinned_ul = torch.empty((fh, fw, 3), dtype=torch.int32, pin_memory=True)
        for rgb_np in batch_frames_np:
            np.copyto(pinned_ul.numpy(), rgb_np.astype(np.int32))
            t_i32 = pinned_ul.cuda(non_blocking=True)
            full_gpu_cache.append(t_i32)
            full_f32 = t_i32.float() / 65535.0                   # (H,W,3) fp32 [0,1]
            full_f32 = full_f32.permute(2, 0, 1).unsqueeze(0)    # (1,3,H,W)
            # Downscale to proxy
            proxy_t = F.interpolate(full_f32, size=(ph, pw), mode="bilinear", align_corners=False)
            proxy_norm = (proxy_t - 0.5) / 0.5  # Normalize for RAUNE: [0,1] → [-1,1]
            proxy_tensors.append(proxy_norm.squeeze(0))
            del full_f32                                          # free fp32, keep int32

        # RAUNE inference + OKLab delta — unchanged from 8-bit path.
        # The proxy normalization produces [-1, 1] range regardless of whether
        # the source was 8-bit or 10-bit, so this section is unaffected.
        proxy_batch = torch.stack(proxy_tensors).cuda()
        del proxy_tensors

        # Cast to model's dtype (passed in from main(), set once at model load)
        if proxy_batch.dtype != model_dtype:
            proxy_batch = proxy_batch.to(model_dtype)

        # RAUNE inference
        if _use_trt:
            raune_out = model.infer(proxy_batch).float()
        else:
            raune_out = model(proxy_batch).float()
        raune_out = ((raune_out + 1.0) / 2.0).clamp(0.0, 1.0)

        # Handle U-Net padding
        rh, rw = raune_out.shape[2], raune_out.shape[3]
        if rh != ph or rw != pw:
            raune_out = F.interpolate(raune_out, size=(ph, pw), mode="bilinear", align_corners=False)

        # Original proxy (un-normalized) — cast back to fp32 for OKLab math
        orig_proxy = (proxy_batch.float() * 0.5 + 0.5).clamp(0.0, 1.0)
        del proxy_batch

        # OKLab deltas at proxy resolution
        raune_lab = rgb_to_lab(raune_out)
        orig_lab = rgb_to_lab(orig_proxy)
        delta_lab = raune_lab - orig_lab
        del raune_out, orig_proxy, raune_lab, orig_lab

        # Upscale deltas to full resolution
        delta_full = F.interpolate(delta_lab, size=(fh, fw), mode="bilinear", align_corners=False)
        del delta_lab

        # Pre-allocate pinned host buffer for download (~3.5× faster than pageable)
        pinned_dl = torch.empty((fh, fw, 3), dtype=torch.int32, pin_memory=True)

        # Apply transfer per frame — reuse GPU cache, no re-upload
        for i in range(n):
            full_t = full_gpu_cache[i].float() / 65535.0
            full_t = full_t.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

            result = transfer_fn(full_t, delta_full[i:i+1])
            del full_t

            # GPU → pinned CPU → uint16 (round-half-up to avoid systematic -0.5 LSB bias)
            gpu_i32 = (result.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 65535.0 + 0.5
                      ).to(torch.int32)
            pinned_dl.copy_(gpu_i32)
            results.append(pinned_dl.numpy().astype(np.uint16))
            del result, gpu_i32

        del full_gpu_cache, delta_full

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RAUNE-Net OKLab chroma transfer")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--full-width", type=int, required=True)
    parser.add_argument("--full-height", type=int, required=True)
    parser.add_argument("--proxy-width", type=int, required=True)
    parser.add_argument("--proxy-height", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Frames per batch (default: 4). Must match grade.rs default.")
    # Single-process mode args
    parser.add_argument("--input", help="Input video file (enables single-process mode)")
    parser.add_argument("--output", help="Output video file (required with --input)")
    parser.add_argument("--output-codec", default="prores_ks",
                        help="Output codec: prores_ks, hevc, libx265, h264 (default: prores_ks)")
    parser.add_argument("--tensorrt", action="store_true",
                        help="Use TensorRT FP16 engine instead of PyTorch (requires tensorrt-cu12)")
    parser.add_argument("--trt-cache-dir", default=None,
                        help="TRT engine cache directory (default: <models-dir>/trt_cache)")
    parser.add_argument("--onnx-path", default=None,
                        help="Pre-exported ONNX model path (default: auto-export to <models-dir>/raune_net.onnx)")
    parser.add_argument("--torch-compile", action="store_true",
                        help="Apply torch.compile to PyTorch model (inductor backend, ~15-30%% speedup)")
    args = parser.parse_args()

    # Load RAUNE model — either TRT engine or PyTorch
    models_dir = Path(args.models_dir)
    if (models_dir / "models" / "raune_net.py").exists():
        raune_dir = models_dir
    elif (models_dir / "models" / "RAUNE-Net" / "models" / "raune_net.py").exists():
        raune_dir = models_dir / "models" / "RAUNE-Net"
    else:
        raune_dir = models_dir
    if str(raune_dir) not in sys.path:
        sys.path.insert(0, str(raune_dir))

    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    _use_trt = args.tensorrt

    if args.tensorrt:
        from dorea_inference.trt_engine import RauneTRTEngine
        from dorea_inference.export_onnx import export_raune_onnx

        # Resolve ONNX path — validate if user-provided
        onnx_path = args.onnx_path
        if onnx_path is not None:
            if not Path(onnx_path).exists():
                print(f"error: --onnx-path does not exist: {onnx_path}", file=sys.stderr)
                sys.exit(1)
        else:
            onnx_path = str(models_dir / "raune_net.onnx")
            if not Path(onnx_path).exists():
                print(f"[raune-filter] Exporting ONNX to {onnx_path}...",
                      file=sys.stderr, flush=True)
                export_raune_onnx(args.weights, str(models_dir), onnx_path)

        # Resolve cache dir and proxy dimensions
        trt_cache_dir = args.trt_cache_dir or str(models_dir / "trt_cache")

        model = RauneTRTEngine.get_or_build(
            onnx_path=onnx_path,
            cache_dir=trt_cache_dir,
            batch_size=args.batch_size,
            height=args.proxy_height,
            width=args.proxy_width,
            fp16=True,
        )
        model_dtype = torch.float16
        print(f"[raune-filter] Using TensorRT FP16 engine", file=sys.stderr, flush=True)
    else:
        from models.raune_net import RauneNet

        model = RauneNet(input_nc=3, output_nc=3, n_blocks=30, n_down=2, ngf=64).cuda()
        state = torch.load(args.weights, map_location="cuda", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        import torch.nn as nn
        model = model.half()
        instance_norm_count = 0
        for m in model.modules():
            if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                m.float()
                instance_norm_count += 1
        model_dtype = next(model.parameters()).dtype
        print(f"[raune-filter] RAUNE converted to fp16 "
              f"({instance_norm_count} InstanceNorm layers kept in fp32, "
              f"model_dtype={model_dtype})",
              file=sys.stderr, flush=True)
        if args.torch_compile:
            model = torch.compile(model, mode="default")
            print("[raune-filter] Applied torch.compile (inductor backend)",
                  file=sys.stderr, flush=True)

    # Single-process mode only (pipe mode removed)
    if not args.input or not args.output:
        print("error: --input and --output are required", file=sys.stderr)
        sys.exit(1)
    run_single_process(args, model, normalize, model_dtype, _use_trt=_use_trt)


if __name__ == "__main__":
    main()
