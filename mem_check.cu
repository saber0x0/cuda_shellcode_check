#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
//#include<yara.h>
/*
在CUDA中，不能直接遍历GPU内存，因为GPU内存管理是由CUDA运行时负责的，而且GPU内存地址对主机（CPU）来说是不透明的。
初始化GPU内存，然后将初始化的数据复制到统一内存区域，再从统一内存复制到主机内存, 后来发现直接读是可以的, 封装一个malloc也可以让它变的透明
//统一内存（Unified Memory）/主机内存映射（Host Memory Mapping）/零拷贝内存（Zero-copy Memory）/直接主机内存访问（Direct Host Memory Access）/页锁定内存（Page-locked Memory）
*/

// CUDA内核copyDataToUnifiedMemory
__global__ void
copyDataToUnifiedMemory(const char* src, char* dst, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

__global__ void 
initializeMemory(int* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = idx;
    }
}
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------//

typedef struct MemoryBlock {
    void* ptr;
    size_t size;
    cudaMemoryType memoryType;
    struct MemoryBlock* next;
} MemoryBlock;

//------------------------------------------------------------------------------------------------------------------------------------------------------------------------//

MemoryBlock* head = NULL;

//------------------------------------------------------------------------------------------------------------------------------------------------------------------------//

void* 
bcMalloc(size_t size, cudaMemoryType memoryType) {
    void* ptr;
    cudaError_t result = cudaMalloc(&ptr, size);
    if (result == cudaSuccess) {
        MemoryBlock* block = (MemoryBlock*)malloc(sizeof(MemoryBlock));
        block->ptr = ptr;
        block->size = size;
        block->memoryType = memoryType;
        block->next = head;
        head = block;
        return ptr;
    }
    else {
        fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(result));
        return NULL;
    }
}
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------//

void 
bcFree(void* ptr) {
    MemoryBlock* current = head;
    MemoryBlock* previous = NULL;
    while (current != NULL) {
        if (current->ptr == ptr) {
            if (previous == NULL) {
                head = current->next;
            }
            else {
                previous->next = current->next;
            }
            cudaFree(ptr);
            free(current);
            return;
        }
        previous = current;
        current = current->next;
    }
    fprintf(stderr, "Attempted to free untracked memory pointer.\n");
}
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------//

size_t 
getAllocatedMemory() {
    MemoryBlock* current = head;
    size_t total = 0;
    while (current != NULL) {
        total += current->size;
        current = current->next;
        // printf("块地址:%d\n块大小:%d\n下一块大小:%d\n", current->ptr, current->size, current->next);
        // 设备内存/主机内存/统一内存/常量内存/纹理内存/表面内存
    }
    return total;
}
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------//

void printMemoryType(void* ptr) {
    cudaPointerAttributes attributes;
    cudaError_t result = cudaPointerGetAttributes(&attributes, ptr);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaPointerGetAttributes failed: %s\n", cudaGetErrorString(result));
        return;
    }
    switch (attributes.type) {
    case cudaMemoryTypeHost:
        printf("内存类型: 主机内存\n");
        break;
    case cudaMemoryTypeDevice:
        printf("内存类型: 设备内存\n");
        break;
    case cudaMemoryTypeManaged:
        printf("内存类型: 统一内存\n");
        break;
    case cudaMemoryTypeUnregistered:
    default:
        printf("未知或未注册的内存类型\n");
        break;
    }
}
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------//

void printAllocatedBlocks() {
    MemoryBlock* current = head;
    while (current != NULL) {
        printf("Memory Block:\n");
        printf("  Pointer: %p\n", current->ptr);
        printf("  Size: %zu\n", current->size);

        // 打印内存类型
        switch (current->memoryType) {
        case cudaMemoryTypeHost:
            printf("  Memory Type: Host\n");
            break;
        case cudaMemoryTypeDevice:
            printf("  Memory Type: Device\n");
            break;
        case cudaMemoryTypeManaged:
            printf("  Memory Type: Managed (Unified)\n");
            break;
        case cudaMemoryTypeUnregistered:
            printf("  Memory Type: Unregistered\n");
            break;
        default:
            printf("  Memory Type: Unknown\n");
            break;
        }
        current = current->next;
    }
}
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------//

int
main() {
    // 转储的内存大小1KB
    const size_t size = 1024;
    size_t totol;
    // 设备指针
    char* d_src, * d_dst, * h_dst, * h_dst2, * h_dst3;
    // 分配了一块统一内存区域d_dst, 这里是统一内存
    cudaMallocManaged(&d_dst, size);
    // 分配了另一块设备内存d_src
    // 这里假设d_src是已知的，指向我们要转储的GPU内存区域
    cudaMalloc(&d_src, size);
    printMemoryType((void*)d_src);


    // 定义线程块和网格大小
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    // 初始化
    // initializeMemory << <numBlocks, blockSize >> > ((int*)d_src, size);
    // 初始化示例
    cudaMemset(d_src, 'A', size);
    printMemoryType((void*)d_dst);

    // 数据复制到统一内存
    copyDataToUnifiedMemory << <numBlocks, blockSize >> > (d_src, d_dst, size);

    // 等待GPU完成
    cudaDeviceSynchronize();

    // 调用内核函数初始化设备内存

    // 将统一内存区域d_dst的内容复制到主机内存h_dst
    h_dst = (char*)malloc(size);
    cudaMemcpy(h_dst, d_dst, size, cudaMemcpyDeviceToHost);
    printMemoryType((void*)h_dst);
    printf("shellcode1:\n%s\n", h_dst);

    cudaFree(d_src);
    cudaFree(d_dst);
    free(h_dst);

    // 封装的cudaMalloc和cudaFree
    void* n_malloc = bcMalloc(size, cudaMemoryTypeDevice);
    cudaMemset(n_malloc, 'B', size);
    h_dst2 = (char*)malloc(size);
    cudaMemcpy(h_dst2, n_malloc, size, cudaMemcpyDeviceToHost);
    printMemoryType(n_malloc);
    printf("shellcode2:\n%s\n", h_dst2);
    //totol = getAllocatedMemory();
    //printf("申请的内存大小:\n%d\n", (int)totol);
    h_dst3 = (char*)malloc(size);
    cudaMemcpy(h_dst3, (char*)0x703000000, size, cudaMemcpyDeviceToHost);
    printf("shellcode22:\n%s\n", h_dst3);
    printAllocatedBlocks();
    bcFree(n_malloc);
    return 0;
}