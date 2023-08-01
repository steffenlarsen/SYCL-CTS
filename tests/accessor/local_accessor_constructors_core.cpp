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
//  Provides sycl::local_accessor constructors test for generic types
//
*******************************************************************************/

#include "../common/common.h"

// FIXME: re-enable when sycl::local_accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL

#include "accessor_common.h"
#include "local_accessor_constructors.h"

using namespace local_accessor_constructors;
#endif

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

namespace local_accessor_constructors_core {

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("sycl::local_accessor constructors. core types", "[accessor]",
 test_combinations)({
  common_run_tests<run_local_constructors_test, TestType>();
});
}  // namespace local_accessor_constructors_core
