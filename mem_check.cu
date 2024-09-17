#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
//#include<yara.h>
/*
��CUDA�У�����ֱ�ӱ���GPU�ڴ棬��ΪGPU�ڴ��������CUDA����ʱ����ģ�����GPU�ڴ��ַ��������CPU����˵�ǲ�͸���ġ�
��ʼ��GPU�ڴ棬Ȼ�󽫳�ʼ�������ݸ��Ƶ�ͳһ�ڴ������ٴ�ͳһ�ڴ渴�Ƶ������ڴ�, ��������ֱ�Ӷ��ǿ��Ե�, ��װһ��mallocҲ�����������͸��
//ͳһ�ڴ棨Unified Memory��/�����ڴ�ӳ�䣨Host Memory Mapping��/�㿽���ڴ棨Zero-copy Memory��/ֱ�������ڴ���ʣ�Direct Host Memory Access��/ҳ�����ڴ棨Page-locked Memory��
*/

// CUDA�ں�copyDataToUnifiedMemory
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
        // printf("���ַ:%d\n���С:%d\n��һ���С:%d\n", current->ptr, current->size, current->next);
        // �豸�ڴ�/�����ڴ�/ͳһ�ڴ�/�����ڴ�/�����ڴ�/�����ڴ�
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
        printf("�ڴ�����: �����ڴ�\n");
        break;
    case cudaMemoryTypeDevice:
        printf("�ڴ�����: �豸�ڴ�\n");
        break;
    case cudaMemoryTypeManaged:
        printf("�ڴ�����: ͳһ�ڴ�\n");
        break;
    case cudaMemoryTypeUnregistered:
    default:
        printf("δ֪��δע����ڴ�����\n");
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

        // ��ӡ�ڴ�����
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
    // ת�����ڴ��С1KB
    const size_t size = 1024;
    size_t totol;
    // �豸ָ��
    char* d_src, * d_dst, * h_dst, * h_dst2, * h_dst3;
    // ������һ��ͳһ�ڴ�����d_dst, ������ͳһ�ڴ�
    cudaMallocManaged(&d_dst, size);
    // ��������һ���豸�ڴ�d_src
    // �������d_src����֪�ģ�ָ������Ҫת����GPU�ڴ�����
    cudaMalloc(&d_src, size);
    printMemoryType((void*)d_src);


    // �����߳̿�������С
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    // ��ʼ��
    // initializeMemory << <numBlocks, blockSize >> > ((int*)d_src, size);
    // ��ʼ��ʾ��
    cudaMemset(d_src, 'A', size);
    printMemoryType((void*)d_dst);

    // ���ݸ��Ƶ�ͳһ�ڴ�
    copyDataToUnifiedMemory << <numBlocks, blockSize >> > (d_src, d_dst, size);

    // �ȴ�GPU���
    cudaDeviceSynchronize();

    // �����ں˺�����ʼ���豸�ڴ�

    // ��ͳһ�ڴ�����d_dst�����ݸ��Ƶ������ڴ�h_dst
    h_dst = (char*)malloc(size);
    cudaMemcpy(h_dst, d_dst, size, cudaMemcpyDeviceToHost);
    printMemoryType((void*)h_dst);
    printf("shellcode1:\n%s\n", h_dst);

    cudaFree(d_src);
    cudaFree(d_dst);
    free(h_dst);

    // ��װ��cudaMalloc��cudaFree
    void* n_malloc = bcMalloc(size, cudaMemoryTypeDevice);
    cudaMemset(n_malloc, 'B', size);
    h_dst2 = (char*)malloc(size);
    cudaMemcpy(h_dst2, n_malloc, size, cudaMemcpyDeviceToHost);
    printMemoryType(n_malloc);
    printf("shellcode2:\n%s\n", h_dst2);
    //totol = getAllocatedMemory();
    //printf("������ڴ��С:\n%d\n", (int)totol);
    h_dst3 = (char*)malloc(size);
    cudaMemcpy(h_dst3, (char*)0x703000000, size, cudaMemcpyDeviceToHost);
    printf("shellcode22:\n%s\n", h_dst3);
    printAllocatedBlocks();
    bcFree(n_malloc);
    return 0;
}