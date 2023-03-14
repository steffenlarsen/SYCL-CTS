/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for device_global arrow operator
//
//  Test executes in the kernel. This test have result array,
//  that kernel uses through a buffer accessor. For result array defined enum
//  class that have to be used for addressing elements in the array. Array
//  with strings used to print error corresponding to the index in result array.
//  This test is provided only for user defined structs that overload
//  operator->()
//
*******************************************************************************/

#include "../../../util/array.h"
#include "../../common/common.h"
#include "../../common/type_list.h"
#include "device_global_common.h"

#define TEST_NAME device_global_api_arrow_operator

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace device_global_common_functions;

#if defined(SYCL_EXT_ONEAPI_PROPERTIES) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
using namespace sycl::ext::oneapi::experimental;

template <typename T, test_names name>
struct device_global_kernel_name;

namespace arrow_operator {
// Creating instance with default constructor
template <typename T>
device_global<T> dev_global;

// Used to address elements in the result array
enum class indx : size_t {
  correct_def_val,
  correct_changed_val,
  size  // must be last
};
constexpr size_t integral(const indx& i) { return to_integral<indx>(i); }

/** @brief The function tests that device_global correctly operate with arrow
 * operator of underlying struct
 *  @tparam T Type of underlying struct
 */
template <typename T>
void run_test(util::logger& log, const std::string& type_name) {
  using kernel = device_global_kernel_name<T, test_names::arrow_operator>;

  std::string error_strings[integral(indx::size)]{
      "Wrong default value returned by arrow operator",
      "Wrong value after change returned by arrow operator",
  };

  util::array<bool, integral(indx::size)> result;
  {
    sycl::buffer<bool, 1> result_buf(result.values,
                                     sycl::range<1>(integral(indx::size)));

    auto queue = util::get_cts_object::queue();
    queue.submit([&](sycl::handler& cgh) {
      auto result_acc =
          result_buf.template get_access<sycl::access_mode::read_write>(cgh);

      T value_ref_zero_init;
      std::memset(&value_ref_zero_init, 0, sizeof(value_ref_zero_init));
      cgh.single_task<kernel>([=] {
        // Check that arrow operator reference to zero-initialized value
        result_acc[integral(indx::correct_def_val)] &=
            (dev_global<T>->a == value_ref_zero_init->a &&
             dev_global<T>->b == value_ref_zero_init->b &&
             dev_global<T>->c == value_ref_zero_init->c);

        T new_values{10, 10, 10};
        // Changing non const values
        dev_global<T>->a = new_values->a;
        dev_global<T>->b = new_values->b;
        dev_global<T>->c = new_values->c;

        // Check, that arrow operator points to new values
        result_acc[integral(indx::correct_changed_val)] =
            (dev_global<T>->a == new_values->a &&
             dev_global<T>->b == new_values->b &&
             dev_global<T>->c == new_values->c);
      });
    });
  }
  for (size_t i = integral(indx::correct_def_val); i < integral(indx::size);
       ++i) {
    if (!result[i]) {
      std::string fail_msg = get_case_description("Device global: operator->()",
                                                  error_strings[i], type_name);
      FAIL(log, fail_msg);
    }
  }
}
}  // namespace arrow_operator
template <typename T>
class check_device_global_api_arrow_operator_for_type {
 public:
  void operator()(sycl_cts::util::logger& log, const std::string& type_name) {
    // Run test for arrow operator overload
    arrow_operator::run_test<T>(log, type_name);
  }
};
#endif

/** test device_global arrow operator
 */
class TEST_NAME : public sycl_cts::util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger& log) override {
#if !defined(SYCL_EXT_ONEAPI_PROPERTIES)
    SKIP("SYCL_EXT_ONEAPI_PROPERTIES is not defined");
#elif !defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
    SKIP("SYCL_EXT_ONEAPI_DEVICE_GLOBAL is not defined");
#else
    check_device_global_api_arrow_operator_for_type<
        user_def_types::arrow_operator_overloaded>{}(log, "arrow_operator");
#endif
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
