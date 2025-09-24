## Build

```
nvcc -rdc=true -o merkle_simplified merkle_simplified.cu fr-tensor.cu keccak.cu bls12-381.cu ioutils.cu -std=c++17 -arch=sm_86
```

## Run

```
./merkle_simplified llama_inference_50q9lauv/
```
