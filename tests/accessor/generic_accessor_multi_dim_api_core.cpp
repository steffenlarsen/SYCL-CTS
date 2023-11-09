/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides generic sycl::accessor api test for generic types
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL

#include "accessor_common.h"
#include "generic_accessor_multi_dim_api.h"

using namespace generic_accessor_multi_dim_api;
#endif

namespace generic_accessor_multi_dim_api_core {

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("Multi-dimensional sycl::accessor api. core types", "[accessor]",
 test_combinations)({
  using namespace generic_accessor_multi_dim_api;
  common_run_tests<run_multi_dim_api_for_type, TestType>();
});

}  // namespace generic_accessor_multi_dim_api_core
