// =================================================================================================
// Simplified Merkle Commitment and Proof Processor
//
// 1. A single root is generated
// 2. A single buffer of size 2*N-1 holds the entire tree.
// 3. Leaves are hashed and placed in the last N slots of the buffer.
// 4. Internal nodes are computed iteratively upwards in a standard binary heap layout.
// 5. The exact Keccak implementation provided is used.
//
// HOW TO COMPILE:
// You will need ioutils.cpp and the associated headers for FrTensor.
// nvcc -rdc=true -o merkle_simplified merkle_simplified.cu fr-tensor.cu keccak.cu bls12-381.cu ioutils.cu -std=c++17 -arch=sm_86
//
// HOW TO RUN:
// ./merkle_simplified /path/to/your/tensor_directory
// =================================================================================================

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <sstream>

// --- Includes for FrTensor library ---
#include "fr-tensor.cuh"
#include "ioutils.cuh"

// --- Basic Types and CUDA Utilities ---
typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;

#define HASH_SIZE_U64 4
#define TPB 256 // A more common thread block size
#define CHECKCUDAERR(err) (HandleCUDAError(err, __FILE__, __LINE__))

static inline void HandleCUDAError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::stringstream ss;
        ss << "CUDA error: " << cudaGetErrorString(err) 
           << " in file " << file 
           << " at line " << line;
        throw std::runtime_error(ss.str());
    }
}

// --- Keccak Implementation https://github.com/okx/zeknox/blob/main/native/keccak/keccak.cu ---
#define KECCAK_ROUNDS 24
#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

// Device-side constants
__constant__ u64 keccakf_rndc[24];
__constant__ int keccakf_rotc[24];
__constant__ int keccakf_piln[24];

// Host-side constants for verification
const u64 KECCAKF_RNDC_HOST[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};
const int KECCAKF_ROTC_HOST[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
const int KECCAKF_PILN_HOST[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};

// Generic Keccak-f permutation, callable from host or device
__host__ __device__ void keccakf(u64 st[25], const u64* rndc, const int* rotc, const int* piln) {
    u64 t, bc[5];
    for (int round = 0; round < KECCAK_ROUNDS; round++) {
        for (int i = 0; i < 5; i++) bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
        }
        t = st[1];
        for (int i = 0; i < 24; i++) {
            int j = piln[i];
            bc[0] = st[j];
            st[j] = ROTL64(t, rotc[i]);
            t = bc[0];
        }
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++) bc[i] = st[j + i];
            for (int i = 0; i < 5; i++) st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        st[0] ^= rndc[round];
    }
}

// Generic Keccak hash function using preprocessor directives
__host__ __device__ void keccak(const u8 *in, int inlen, u8 *md, int mdlen) {
    u64 st[25];
    u8 temp[144];
    int rsiz = 200 - 2 * mdlen;
    int rsizw = rsiz / 8;
    memset(st, 0, sizeof(st));
    for ( ; inlen >= rsiz; inlen -= rsiz, in += rsiz) {
        for (int i = 0; i < rsizw; i++) st[i] ^= ((u64 *) in)[i];
        #if defined(__CUDA_ARCH__)
            keccakf(st, keccakf_rndc, keccakf_rotc, keccakf_piln);
        #else
            keccakf(st, KECCAKF_RNDC_HOST, KECCAKF_ROTC_HOST, KECCAKF_PILN_HOST);
        #endif
    }
    memcpy(temp, in, inlen);
    temp[inlen++] = 1;
    memset(temp + inlen, 0, rsiz - inlen);
    temp[rsiz - 1] |= 0x80;
    for (int i = 0; i < rsizw; i++) st[i] ^= ((u64 *) temp)[i];
    #if defined(__CUDA_ARCH__)
        keccakf(st, keccakf_rndc, keccakf_rotc, keccakf_piln);
    #else
        keccakf(st, KECCAKF_RNDC_HOST, KECCAKF_ROTC_HOST, KECCAKF_PILN_HOST);
    #endif
    memcpy(md, st, mdlen);
}

// Wrapper class for hashing operations
class KeccakHasher {
public:
    // Hashes a single leaf (Fr_t element)
    __device__ static void hash_one(const Fr_t* leaf, u64* hash_out) {
        keccak((const u8*)leaf, sizeof(Fr_t), (u8*)hash_out, 32);
        hash_out[3] &= 0xFF; // Truncate to 25 bytes
    }

    // Hashes two 25-byte intermediate hashes
    __device__ static void hash_two(const u64* hash1, const u64* hash2, u64* hash_out) {
        u8 input[50];
        memcpy(input, hash1, 25);
        memcpy(input + 25, hash2, 25);
        keccak(input, 50, (u8*)hash_out, 32);
        hash_out[3] &= 0xFF; // Truncate to 25 bytes
    }
};

// Kernel 1: Hashes all leaves and places them at the end of the digest buffer.
// Layout: [Internal Nodes (N-1)] [Leaf Hashes (N)]
__global__ void hash_leaves_kernel(const Fr_t* leaves, u64 num_leaves, u64* digests_out) {
    u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_leaves) return;

    // The first (num_leaves - 1) slots are for internal nodes. Leaves start after that.
    u64 internal_node_count = num_leaves - 1;
    u64 digest_idx = internal_node_count + tid;

    KeccakHasher::hash_one(&leaves[tid], &digests_out[digest_idx * HASH_SIZE_U64]);
}

// Kernel 2: Builds one level of the Merkle tree.
__global__ void build_tree_level_kernel(u64* digests, u32 level_start_idx, u32 level_node_count) {
    u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_node_count) return;

    // Absolute index of the parent node we are computing in this level
    u32 parent_abs_idx = level_start_idx + tid;
    
    // Standard binary heap child calculation (0-indexed)
    u32 child1_abs_idx = 2 * parent_abs_idx + 1;
    u32 child2_abs_idx = 2 * parent_abs_idx + 2;

    u64* parent_ptr = &digests[parent_abs_idx * HASH_SIZE_U64];
    u64* child1_ptr = &digests[child1_abs_idx * HASH_SIZE_U64];
    u64* child2_ptr = &digests[child2_abs_idx * HASH_SIZE_U64];

    KeccakHasher::hash_two(child1_ptr, child2_ptr, parent_ptr);
}

// Proof generation and verification

struct MerkleProof {
    std::vector<std::vector<u64>> siblings;
};

// hash two on host
void hash_two_on_host(const std::vector<u64>& h1, const std::vector<u64>& h2, std::vector<u64>& result) {
    u8 input[50];
    memcpy(input, h1.data(), 25);
    memcpy(input + 25, h2.data(), 25);
    result.resize(HASH_SIZE_U64);
    keccak(input, 50, (u8*)result.data(), 32);
    result[3] &= 0xFF; // Ensure truncation matches device
}

// Generates a Merkle proof for a given leaf index.
MerkleProof generate_merkle_proof(const std::vector<u64>& h_digests, u64 num_leaves, u32 leaf_index) {
    MerkleProof proof;
    u64 tree_height = (num_leaves > 0) ? (u64)log2(num_leaves) : 0;
    
    // Start at the leaf's absolute index in the digest buffer
    u64 current_idx = (num_leaves - 1) + leaf_index;

    for (u64 level = 0; level < tree_height; ++level) {
        // Determine sibling's index using standard heap logic
        u64 sibling_idx = (current_idx % 2 != 0) ? current_idx + 1 : current_idx - 1;
        
        std::vector<u64> sibling_hash(HASH_SIZE_U64);
        const u64* sibling_ptr = &h_digests[sibling_idx * HASH_SIZE_U64];
        memcpy(sibling_hash.data(), sibling_ptr, HASH_SIZE_U64 * sizeof(u64));
        proof.siblings.push_back(sibling_hash);

        // Move up to the parent node
        current_idx = (current_idx - 1) / 2;
    }
    return proof;
}

// Verifies a Merkle proof.
bool verify_merkle_proof(std::vector<u64>& leaf_hash, const MerkleProof& proof, const std::vector<u64>& root, u32 leaf_index) {
    std::vector<u64> current_hash = leaf_hash;
    
    for (const auto& sibling_hash : proof.siblings) {
        std::vector<u64> next_hash;
        if (leaf_index % 2 == 0) { // Current path node is a left child
            hash_two_on_host(current_hash, sibling_hash, next_hash);
        } else { // Current path node is a right child
            hash_two_on_host(sibling_hash, current_hash, next_hash);
        }
        current_hash = next_hash;
        leaf_index /= 2; // Move up to the parent's perspective
    }
    return current_hash == root;
}

// struct to hold the results of the commitment process
struct CommitResult {
    std::string filename;
    double commit_time_ms;
    double proof_gen_time_ms;
    size_t proof_size_bytes;
    std::vector<u64> root;
    std::string verification_status;
    std::string error_message;
};

// Orchestrates the entire commitment process on the GPU.
void compute_merkle_commitment(const FrTensor& tensor, std::vector<u64>& h_root, std::vector<u64>& h_digests) {
    u64 num_leaves = tensor.size;
    if (num_leaves == 0) {
        h_root.assign(HASH_SIZE_U64, 0);
        return;
    }
    if ((num_leaves & (num_leaves - 1)) != 0) {
        throw std::runtime_error("Tensor size must be a power of 2.");
    }

    u64 digest_buf_len = (2 * num_leaves) - 1;
    h_digests.resize(digest_buf_len * HASH_SIZE_U64);

    Fr_t* d_leaves = nullptr;
    u64* d_digests = nullptr;

    CHECKCUDAERR(cudaMalloc(&d_leaves, num_leaves * sizeof(Fr_t)));
    CHECKCUDAERR(cudaMalloc(&d_digests, digest_buf_len * HASH_SIZE_U64 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(d_leaves, tensor.gpu_data, num_leaves * sizeof(Fr_t), cudaMemcpyHostToDevice));

    // 1. Hash all leaves.
    hash_leaves_kernel<<<(num_leaves + TPB - 1) / TPB, TPB>>>(d_leaves, num_leaves, d_digests);

    // 2. Build internal tree levels from bottom up.
    u32 num_levels = (u32)log2(num_leaves);
    for (int level = num_levels - 1; level >= 0; --level) {
        u32 level_start_idx = (1 << level) - 1;
        u32 level_node_count = (1 << level);
        build_tree_level_kernel<<<(level_node_count + TPB - 1) / TPB, TPB>>>(d_digests, level_start_idx, level_node_count);
    }
    
    CHECKCUDAERR(cudaDeviceSynchronize());

    // 3. Copy the full digest buffer back to the host for proof generation.
    CHECKCUDAERR(cudaMemcpy(h_digests.data(), d_digests, h_digests.size() * sizeof(u64), cudaMemcpyDeviceToHost));

    // 4. The root is the first element in the host buffer.
    h_root.resize(HASH_SIZE_U64);
    memcpy(h_root.data(), h_digests.data(), HASH_SIZE_U64 * sizeof(u64));

    // Cleanup
    CHECKCUDAERR(cudaFree(d_leaves));
    CHECKCUDAERR(cudaFree(d_digests));
}


int main(int argc, char* argv[]) {
    // Initialize Keccak constants on the device
    CHECKCUDAERR(cudaMemcpyToSymbol(keccakf_rndc, KECCAKF_RNDC_HOST, sizeof(KECCAKF_RNDC_HOST)));
    CHECKCUDAERR(cudaMemcpyToSymbol(keccakf_rotc, KECCAKF_ROTC_HOST, sizeof(KECCAKF_ROTC_HOST)));
    CHECKCUDAERR(cudaMemcpyToSymbol(keccakf_piln, KECCAKF_PILN_HOST, sizeof(KECCAKF_PILN_HOST)));

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
        return 1;
    }
    std::filesystem::path dir_path(argv[1]);
    if (!std::filesystem::is_directory(dir_path)) {
        std::cerr << "Error: Provided path is not a directory: " << argv[1] << std::endl;
        return 1;
    }

    std::vector<CommitResult> results;
    std::cout << "Processing .bin files in directory: " << dir_path << "\n" << std::endl;

    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            CommitResult result;
            result.filename = entry.path().filename().string();
            
            try {
                FrTensor tensor = FrTensor::from_int_bin(entry.path().string());
                std::vector<u64> h_digests;

                // --- 1. Compute Commitment ---
                auto start_commit = std::chrono::high_resolution_clock::now();
                compute_merkle_commitment(tensor, result.root, h_digests);
                auto end_commit = std::chrono::high_resolution_clock::now();
                result.commit_time_ms = std::chrono::duration<double, std::milli>(end_commit - start_commit).count();

                // --- 2. Generate Proof for leaf 0 ---
                const u32 leaf_to_prove = 0;
                auto start_proof = std::chrono::high_resolution_clock::now();
                MerkleProof proof = generate_merkle_proof(h_digests, tensor.size, leaf_to_prove);
                auto end_proof = std::chrono::high_resolution_clock::now();
                result.proof_gen_time_ms = std::chrono::duration<double, std::milli>(end_proof - start_proof).count();
                result.proof_size_bytes = proof.siblings.size() * HASH_SIZE_U64 * sizeof(u64);

                // --- 3. Verify Proof ---
                std::vector<u64> leaf_hash(HASH_SIZE_U64);
                u64 leaf_hash_idx = (tensor.size - 1 + leaf_to_prove) * HASH_SIZE_U64;
                memcpy(leaf_hash.data(), &h_digests[leaf_hash_idx], HASH_SIZE_U64 * sizeof(u64));
                
                bool is_valid = verify_merkle_proof(leaf_hash, proof, result.root, leaf_to_prove);
                result.verification_status = is_valid ? "Valid" : "INVALID";
                result.error_message = "Success";

            } catch (const std::exception& e) {
                result.commit_time_ms = 0.0;
                result.proof_gen_time_ms = 0.0;
                result.proof_size_bytes = 0;
                result.verification_status = "N/A";
                result.error_message = e.what();
                result.root.assign(HASH_SIZE_U64, 0);
            }
            results.push_back(result);
        }
    }

    // --- Generate Final Report ---
    std::cout << "\n--- Simplified Merkle Report (for leaf 0) ---\n" << std::endl;
    std::cout << std::left 
              << std::setw(35) << "Filename"
              << std::setw(15) << "Commit(ms)"
              << std::setw(15) << "ProofGen(ms)"
              << std::setw(15) << "Proof(bytes)"
              << std::setw(12) << "Verified"
              << std::setw(70) << "Merkle Root (Commitment)"
              << "Status" << std::endl;
    std::cout << std::string(175, '-') << std::endl;

    for (const auto& res : results) {
        std::cout << std::left << std::setw(35) << res.filename
                  << std::fixed << std::setprecision(3) << std::setw(15) << res.commit_time_ms
                  << std::fixed << std::setprecision(3) << std::setw(15) << res.proof_gen_time_ms
                  << std::left << std::setw(15) << res.proof_size_bytes
                  << std::setw(12) << res.verification_status;

        if (res.error_message == "Success") {
            std::stringstream ss;
            ss << "0x";
            for (const auto& val : res.root) {
                ss << std::hex << std::setw(16) << std::setfill('0') << val;
            }
            std::cout << std::left << std::setw(70) << ss.str() << res.error_message << std::endl;
        } else {
            std::cout << std::left << std::setw(70) << "N/A" << "Error: " << res.error_message << std::endl;
        }
    }
    return 0;
}