/-
Copyright (c) 2024 SciLean contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: SciLean contributors
-/
import Lean

-- GPU Tracing Infrastructure
-- Hooks into Lean's compiler tracer for GPU operation observability.
-- Enable tracing with: set_option trace.GPU.metal true

namespace SciLean.GPU

open Lean

-- Register trace classes
initialize registerTraceClass `GPU.metal
initialize registerTraceClass `GPU.metal.alloc
initialize registerTraceClass `GPU.metal.transfer
initialize registerTraceClass `GPU.metal.compute
initialize registerTraceClass `GPU.metal.batch
initialize registerTraceClass `GPU.metal.timing

/-- Get current time in milliseconds for timing -/
def nowMs : IO Nat := IO.monoMsNow

/-- Time an IO operation and trace the result -/
def timeIO (label : String) (op : IO α) : IO α := do
  let t0 ← nowMs
  let result ← op
  let t1 ← nowMs
  let elapsed := t1 - t0
  if elapsed > 0 then
    IO.println s!"[GPU.timing] {label}: {elapsed}ms"
  return result

/-- Format bytes for human-readable output -/
def formatBytes (bytes : Nat) : String :=
  if bytes < 1024 then s!"{bytes} B"
  else if bytes < 1024 * 1024 then s!"{bytes / 1024} KB"
  else if bytes < 1024 * 1024 * 1024 then s!"{bytes / (1024 * 1024)} MB"
  else s!"{bytes / (1024 * 1024 * 1024)} GB"

/-- Format matrix dimensions -/
def formatDims (m k n : Nat) : String := s!"{m}x{k}x{n}"

/-- Trace buffer allocation -/
def traceAlloc (bytes : Nat) : IO Unit := do
  IO.println s!"[GPU.alloc] Allocated {formatBytes bytes}"

/-- Trace data transfer -/
def traceTransfer (direction : String) (bytes : Nat) : IO Unit := do
  IO.println s!"[GPU.transfer] {direction} {formatBytes bytes}"

/-- Trace compute operation -/
def traceCompute (op : String) (details : String := "") : IO Unit := do
  let msg := if details.isEmpty then op else s!"{op} ({details})"
  IO.println s!"[GPU.compute] {msg}"

/-- Trace GEMM operation -/
def traceGemm (m k n : USize) (transA transB : Bool) : IO Unit := do
  let tA := if transA then "T" else "N"
  let tB := if transB then "T" else "N"
  IO.println s!"[GPU.compute] GEMM {tA}{tB} {m}x{k}x{n}"

end SciLean.GPU
