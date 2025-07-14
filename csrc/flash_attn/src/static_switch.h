// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define MAXPAGES_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                          \
    if (COND <= 32) {                            \
      constexpr static int CONST_NAME = 32;  \
      return __VA_ARGS__();                \
    } else if (COND <= 64) {            \
      constexpr static int CONST_NAME = 64;  \
      return __VA_ARGS__();                \
    } else if (COND <= 96) {            \
      constexpr static int CONST_NAME = 96;  \
      return __VA_ARGS__();                \
    } else if (COND <= 128) {           \
      constexpr static int CONST_NAME = 128; \
      return __VA_ARGS__();                \
    } else if (COND <= 160) {           \
      constexpr static int CONST_NAME = 160; \
      return __VA_ARGS__();                \
    } else if (COND <= 192) {           \
      constexpr static int CONST_NAME = 192; \
      return __VA_ARGS__();                \
    } else if (COND <= 224) {           \
      constexpr static int CONST_NAME = 224; \
      return __VA_ARGS__();                \
    } else if (COND <= 256) {           \
      constexpr static int CONST_NAME = 256; \
      return __VA_ARGS__();                \
    } else if (COND <= 288) {           \
      constexpr static int CONST_NAME = 288; \
      return __VA_ARGS__();                \
    } else if (COND <= 320) {           \
      constexpr static int CONST_NAME = 320; \
      return __VA_ARGS__();                \
    } else if (COND <= 352) {           \
      constexpr static int CONST_NAME = 352; \
      return __VA_ARGS__();                \
    } else if (COND <= 384) {           \
      constexpr static int CONST_NAME = 384; \
      return __VA_ARGS__();                \
    } else if (COND <=  416) {           \
      constexpr static int CONST_NAME = 416; \
      return __VA_ARGS__();                \
    } else if (COND <= 448) {           \
      constexpr static int CONST_NAME = 448; \
      return __VA_ARGS__();                \
    } else if (COND <= 480) {           \
      constexpr static int CONST_NAME = 480; \
      return __VA_ARGS__();                \
    } else if (COND <= 512) {           \
      constexpr static int CONST_NAME = 512; \
      return __VA_ARGS__();                \
    } else {                                    \
      constexpr static int CONST_NAME = 1024; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#ifdef FLASHATTENTION_DISABLE_DROPOUT
  #define DROPOUT_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define DROPOUT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_ALIBI
  #define ALIBI_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define ALIBI_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  #define EVENK_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = true;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define EVENK_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  #define SOFTCAP_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define SOFTCAP_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_LOCAL
  #define LOCAL_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define LOCAL_SWITCH BOOL_SWITCH
#endif

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = cutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = cutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

#define HEADDIM_SWITCH(HEADDIM, ...)     \
  [&] {                                    \
    if (HEADDIM <= 32) {                   \
      constexpr static int kHeadDim = 32;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 64) {            \
      constexpr static int kHeadDim = 64;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 96) {            \
      constexpr static int kHeadDim = 96;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 128) {           \
      constexpr static int kHeadDim = 128; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 160) {           \
      constexpr static int kHeadDim = 160; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 192) {           \
      constexpr static int kHeadDim = 192; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 256) {           \
      constexpr static int kHeadDim = 256; \
      return __VA_ARGS__();                \
    }                                      \
  }()