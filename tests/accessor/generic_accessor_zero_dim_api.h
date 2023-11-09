/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#ifndef SYCL_CTS_GENERIC_ACCESSOR_API_COMMON_H
#define SYCL_CTS_GENERIC_ACCESSOR_API_COMMON_H
#include "accessor_common.h"
#include <cmath>

namespace generic_accessor_zero_dim_api {
using namespace sycl_cts;
using namespace accessor_tests_common;

template <typename T, typename AccessT, typename TargetT>
class kernel_buffer_accessor;

template <typename T, typename AccessT, typename TargetT>
class run_api_tests {
  static constexpr sycl::access_mode AccessMode = AccessT::value;
  static constexpr sycl::target Target = TargetT::value;
  using AccT = sycl::accessor<T, 0, AccessMode, Target>;

 public:
  void operator()(const std::string& type_name,
                  const std::string& access_mode_name,
                  const std::string& target_name) {
    auto queue = once_per_unit::get_queue();
    auto r = util::get_cts_object::range<1>::get(1, 1, 1);

    SECTION(get_section_name<0>(type_name, access_mode_name, target_name,
                                "Check api for buffer accessor")) {
      T data = value_operations::init<T>(expected_val);
      constexpr size_t res_size = 15;
      bool res[res_size];
      std::fill(res, res + res_size, true);
      {
        sycl::buffer<T, 1> data_buf(&data, r);
        sycl::buffer res_buf(res, sycl::range(res_size));
        queue
            .submit([&](sycl::handler& cgh) {
              AccT acc(data_buf, cgh);

              if constexpr (Target == sycl::target::host_task) {
                cgh.host_task([=] {
                  T some_data = value_operations::init<T>(expected_val);
                  typename AccT::reference dref = acc;
                  CHECK(value_operations::are_equal(some_data, dref));
                  if constexpr (AccessMode != sycl::access_mode::read) {
                    typename AccT::value_type v_data =
                        value_operations::init<typename AccT::value_type>(
                            changed_val);
                    // check method const AccT::operator=(const T& data) const
                    acc = v_data;
                    CHECK(value_operations::are_equal(dref, v_data));

                    // check method const AccT::operator=(T&& data) const
                    acc = value_operations::init<typename AccT::value_type>(
                        changed_val);
                    CHECK(value_operations::are_equal(dref, v_data));
                  }
                });
              } else {
                using kname = kernel_buffer_accessor<T, AccessT, TargetT>;
                sycl::accessor res_acc(res_buf, cgh);
                cgh.single_task<kname>([acc, res_acc]() {
                  T converted_expected_val =
                      value_operations::init<T>(expected_val);
                  size_t res_i = 0;
                  T some_data = value_operations::init<T>(expected_val);
                  typename AccT::reference dref = acc;
                  res_acc[res_i++] =
                      value_operations::are_equal(some_data, dref);

                  if constexpr (AccessMode != sycl::access_mode::read) {
                    typename AccT::value_type v_data =
                        value_operations::init<typename AccT::value_type>(
                            changed_val);
                    // check method const AccT::operator=(const T& data) const
                    acc = v_data;
                    res_acc[res_i++] =
                        value_operations::are_equal(dref, v_data);

                    // check method const AccT::operator=(T&& data) const
                    acc = value_operations::init<typename AccT::value_type>(
                        changed_val);
                    res_acc[res_i++] =
                        value_operations::are_equal(dref, v_data);
                  }
                });
              }
            })
            .wait_and_throw();
      }
      if constexpr (Target == sycl::target::device) {
        for (size_t i = 0; i < res_size; ++i) {
          CHECK(res[i]);
        }
      }
      if constexpr (AccessMode != sycl::access_mode::read)
        CHECK(value_operations::are_equal(data, changed_val));
    }
  }
};

using test_combinations =
    typename get_combinations<access_modes_pack, targets_pack>::type;

template <typename T, typename ArgCombination>
class run_zero_dim_api_for_type {
 public:
  void operator()(const std::string& type_name) {
    // Get the packs from the test combination type.
    using AccessModePack = std::tuple_element_t<0, ArgCombination>;
    using TargetsPack = std::tuple_element_t<1, ArgCombination>;

    // Type packs instances have to be const, otherwise for_all_combination
    // will not compile
    const auto access_modes = AccessModePack::generate_named();
    const auto targets = TargetsPack::generate_named();

    // To handle cases when class was called from functions
    // like for_all_types_vectors_marray or
    // for_all_device_copyable_std_containers. This will wrap string with type T
    // to string with container<T> if T is an array or other kind of container.
    auto actual_type_name = type_name_string<T>::get(type_name);

    for_all_combinations<run_api_tests, T>(access_modes, targets,
                                           actual_type_name);

    // For covering const types
    actual_type_name = std::string("const ") + actual_type_name;
    // const T can be only with access_mode::read
    const auto read_only_acc_mode =
        value_pack<sycl::access_mode, sycl::access_mode::read>::generate_named(
            "access_mode::read");
    for_all_combinations<run_api_tests, const T>(read_only_acc_mode, targets,
                                                 actual_type_name);
  }
};
}  // namespace generic_accessor_zero_dim_api
#endif  // SYCL_CTS_GENERIC_ACCESSOR_API_COMMON_H
