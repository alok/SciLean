import SciLean.Modules.ML.Jaxpr.AST
import SciLean.Modules.ML.Jaxpr.Elab
import SciLean.Modules.ML.Jaxpr.Metal

open SciLean.ML.Jaxpr

/-!
# Tests for Jaxpr → Metal code generation

Verifies kernel fusion generates correct Metal shader code.
-/

-- Test 1: Simple add kernel
def testAdd : Jaxpr := [jaxpr|
in x y
let z := add x y
out z
]

#check testAdd
#eval testAdd.toString

#eval do
  let result ← IO.ofExcept (jaxprToMetal "test_add" testAdd)
  IO.println "=== Test 1: Simple add ==="
  IO.println result

-- Test 2: Fused relu + add
def testReluAdd : Jaxpr := [jaxpr|
in x y
let z := relu x
let w := add z y
out w
]

#eval do
  let result ← IO.ofExcept (jaxprToMetal "fused_relu_add" testReluAdd)
  IO.println "=== Test 2: Fused relu+add ==="
  IO.println result

-- Test 3: Multiple operations (sigmoid → mul → add)
def testSigmoidMulAdd : Jaxpr := [jaxpr|
in x y z
let s := sigmoid x
let m := mul s y
let r := add m z
out r
]

#eval do
  let result ← IO.ofExcept (jaxprToMetal "fused_sigmoid_mul_add" testSigmoidMulAdd)
  IO.println "=== Test 3: Fused sigmoid→mul→add ==="
  IO.println result

-- Test 4: GELU activation
def testGelu : Jaxpr := [jaxpr|
in x
let y := gelu x
out y
]

#eval do
  let result ← IO.ofExcept (jaxprToMetal "gelu_kernel" testGelu)
  IO.println "=== Test 4: GELU activation ==="
  IO.println result

-- Test 5: Complex fusion chain
def testComplexFusion : Jaxpr := [jaxpr|
in a b
let t1 := mul a b
let t2 := relu t1
let t3 := sigmoid t2
let t4 := add t3 a
out t4
]

#eval do
  let result ← IO.ofExcept (jaxprToMetal "complex_fusion" testComplexFusion)
  IO.println "=== Test 5: Complex fusion (mul→relu→sigmoid→add) ==="
  IO.println result

-- Test 6: Verify fusibility check
#eval do
  IO.println "=== Test 6: Fusibility check ==="
  IO.println s!"testAdd fusible: {isFusible testAdd}"
  IO.println s!"testReluAdd fusible: {isFusible testReluAdd}"
  IO.println s!"testComplexFusion fusible: {isFusible testComplexFusion}"

-- Test 7: Generate full shader file with header
#eval do
  let result ← IO.ofExcept (jaxprToMetalFile "example_kernel" testReluAdd)
  IO.println "=== Test 7: Full shader file ==="
  IO.println result

-- Test 8: Batch multiple kernels
#eval do
  let kernels := [
    ("add_kernel", testAdd),
    ("relu_add_kernel", testReluAdd),
    ("gelu_kernel", testGelu)
  ]
  let result ← IO.ofExcept (jaxprsToMetalFile kernels)
  IO.println "=== Test 8: Batch kernel file ==="
  IO.println result
