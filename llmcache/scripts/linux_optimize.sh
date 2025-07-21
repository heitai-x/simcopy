#!/bin/bash
# Linux系统级共享内存优化脚本

echo "正在优化Linux系统共享内存性能..."

# 1. 配置大页内存
echo "配置大页内存..."
sudo sysctl vm.nr_hugepages=128
echo 128 | sudo tee /proc/sys/vm/nr_hugepages

# 2. 优化共享内存限制
echo "优化共享内存限制..."
sudo sysctl kernel.shmmax=134217728  # 128MB
sudo sysctl kernel.shmall=32768

# 3. 配置内存映射限制
echo "配置内存映射限制..."
sudo sysctl vm.max_map_count=262144

# 4. 优化内存回收策略
echo "优化内存回收策略..."
sudo sysctl vm.swappiness=10
sudo sysctl vm.vfs_cache_pressure=50

# 5. 配置透明大页
echo "配置透明大页..."
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag

# 6. 设置NUMA策略（如果是多NUMA节点系统）
if [ -d "/sys/devices/system/node/node1" ]; then
    echo "检测到NUMA系统，配置内存策略..."
    echo 2 | sudo tee /proc/sys/kernel/numa_balancing
fi

# 7. 创建共享内存目录并设置权限
echo "配置共享内存目录..."
sudo mkdir -p /dev/shm/vllm
sudo chmod 755 /dev/shm/vllm

echo "Linux系统优化完成！"
echo "建议重启应用程序以使配置生效。"