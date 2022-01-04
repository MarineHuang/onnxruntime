// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <memory>
#include <iostream>
#include <memory_resource>
#include <core/common/safeint.h>

#include "core/framework/allocator.h"

#pragma warning(push)
#pragma warning(disable : 4127)
#include <absl/container/inlined_vector.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/flat_hash_map.h>
#pragma warning(pop)

namespace onnxruntime {

template <typename T, size_t N>
using InlinedVector = absl::InlinedVector<T, N>;

template <typename T>
using InlinedHashSet = absl::flat_hash_set<T>;

template <typename K, typename V>
using InlinedHashMap = absl::flat_hash_map<K, V>;

namespace pmr {
template <typename T, size_t N>
using InlinedVector = absl::InlinedVector<T, N, std::pmr::polymorphic_allocator<T>>;

template <typename T, typename Hash = absl::container_internal::hash_default_hash<T>, typename Eq = absl::container_internal::hash_default_eq<T>>
using InlinedHashSet = absl::flat_hash_set<T, Hash, Eq, std::pmr::polymorphic_allocator<T>>;

template <typename K, typename V,
          typename Hash = absl::container_internal::hash_default_hash<K>,
          typename Eq = absl::container_internal::hash_default_eq<K>>
using InlinedHashMap = absl::flat_hash_map<K, V, Hash, Eq, std::pmr::polymorphic_allocator<std::pair<const K, V>>>;
}  // namespace pmr

#ifdef _MSC_VER
#define ORT_ALLOCA(s) _alloca(s)
constexpr size_t kOrtStackAllocationLimitBytes = 4 * 1024;
#elif defined(__GNUC__) || defined(__clang__)
#define ORT_ALLOCA(s) alloca(s)
constexpr size_t kOrtStackAllocationLimitBytes = 4 * 1024;
#else
// always on the heap
#define ORT_ALLOCA(s) nullptr
constexpr size_t kOrtStackAllocationLimitBytes = 0;
#endif

namespace inline_containers_internal {
inline void* allocate(size_t size, std::unique_ptr<uint8_t[]>& buf) {
  buf = std::make_unique<uint8_t[]>(size);
  return buf.get();
}

inline size_t EstimateHashStorageSize(size_t slot_size, size_t num_elements) {
  // See https://abseil.io/docs/cpp/guides/container#memory-usage
  // However, the picture is a lot more complex
  // up to a power of two - 1 with minimum of 1
  constexpr size_t num_cloned_bytes = 15;

  const SafeInt<size_t> nelem = num_elements ? ~size_t{} >> absl::countl_zero(num_elements) : 1;
  const SafeInt<size_t> num_control_bytes = nelem + 1 + num_cloned_bytes;
  const SafeInt<size_t> slot_offset = (num_control_bytes + slot_size - 1) & (~slot_size + 1);
  return (slot_offset + nelem * slot_size);
}

inline void* allocate_and_align(AllocatorPtr allocator, size_t size, size_t alignment,
                                IAllocatorUniquePtr<void>& buf) {
  size_t to_allocate;
  bool result = IAllocator::CalcMemSizeForArrayWithAlignment(size, sizeof(uint8_t), alignment, &to_allocate);
  if (!result) {
    return nullptr;
  }
  buf = IAllocator::MakeUniquePtr<void>(std::move(allocator), to_allocate);
  void* ptr = buf.get();
  return std::align(alignment, to_allocate, ptr, to_allocate);
}
}  // namespace inline_containers_internal

/// <summary>
/// Estimate memory requirements for an InlinedHashSet
/// so it can be pre-allocated on a stack or using other allocator when the number
/// of elements is known. This provides an oppty to bring the number of allocations
/// down to zero.
/// </summary>
/// <param name="value_size">sizeof(Cont::value_type)</param>
/// <param name="num_elements">number of elements</param>
/// <returns></returns>
template <class T>
inline size_t EstimateInlinedHashSetMemory(size_t num_elements) {
  constexpr size_t slot_size = sizeof(InlinedHashSet<T>::slot_type);
  return inline_containers_internal::EstimateHashStorageSize(slot_size, num_elements);
}

template <class K, class V>
inline size_t EstimateInlinedHashMapMemory(size_t num_elements) {
  constexpr size_t slot_size = sizeof(InlinedHashMap<K, V>::slot_type);
  return inline_containers_internal::EstimateHashStorageSize(slot_size, num_elements);
}

inline bool IsSizeOverStackAllocationLimit(size_t size) {
  return size > kOrtStackAllocationLimitBytes;
}

#define OrtDeclareAllignedStackOrAllocatedBuffer(buffer_ptr, size_in_bytes)                          \
  std::unique_ptr<uint8_t[]> on_heap_##buffer_ptr;                                                   \
  void* buffer_ptr = (size_in_bytes > kOrtStackAllocationLimitBytes)                                 \
                         ? inline_containers_internal::allocate(size_in_bytes, on_heap_##buffer_ptr) \
                         : ORT_ALLOCA(size_in_bytes)

// This gives a set size stackbuffer
template <typename T, size_t N>
class SmallBuffer {
  T buffer_[N];

 public:
  T* Buffer() noexcept { return buffer_; }
  constexpr size_t size() const noexcept { return N; }
  constexpr size_t size_in_bytes() const noexcept { return sizeof(T) * N; }
};

class SmallBufferResource {
  std::pmr::monotonic_buffer_resource resource_;

 public:
  SmallBufferResource(void* ptr, size_t size_in_bytes)
      : resource_(ptr, size_in_bytes, std::pmr::get_default_resource()) {}
  SmallBufferResource(void* ptr, size_t size_in_bytes, std::pmr::memory_resource* upstream)
      : resource_(ptr, size_in_bytes, upstream) {}
  std::pmr::memory_resource* resource() noexcept { return &resource_; }
  std::pmr::memory_resource* upstream() const noexcept { return resource_.upstream_resource(); }
};

class DebugMemoryResource : public std::pmr::memory_resource {
 public:
  DebugMemoryResource(std::string name, std::pmr::memory_resource* up)
      : name_(std::move(name)), upstream_(up) {}

  size_t Allocated() const noexcept { return total_allocated_; }
  size_t Deallocated() const noexcept { return total_deallocated_; }

 private:
  std::string name_;
  std::pmr::memory_resource* upstream_;
  size_t total_allocated_ = 0;
  size_t total_deallocated_ = 0;

 private:
  void* do_allocate(size_t bytes, size_t align) override {
    std::cout << name_ << " : allocate : " << bytes << std::endl;
    total_allocated_ += bytes;
    return upstream_->allocate(bytes, align);
  }
  void do_deallocate(void* ptr, size_t bytes, size_t align) override {
    std::cout << name_ << " : deallocate : " << bytes << std::endl;
    total_deallocated_ += bytes;
    upstream_->deallocate(ptr, bytes, align);
  }
  bool do_is_equal(const memory_resource&) const noexcept override { return false; }
};

}  // namespace onnxruntime
