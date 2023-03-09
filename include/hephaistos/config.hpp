#pragma once

#if !defined(HEPHAISTOS_STATIC)
#   if defined(_WIN32) || defined(__CYGWIN__)
#       define HEPHAISTOS_API_EXPORT __declspec(dllexport)
#       define HEPHAISTOS_API_IMPORT __declspec(dllimport)
#   else
#       define HEPHAISTOS_API_EXPORT __attribute__ (("default"))
#       define HEPHAISTOS_API_IMPORT __attribute__ (("default"))
#   endif
#else
#   define HEPHAISTOS_API_EXPORT
#   define HEPHAISTOS_API_IMPORT
#endif

//import/export switch
#if defined(HEPHAISTOS_EXPORTS)
#   define HEPHAISTOS_API HEPHAISTOS_API_EXPORT
#else
#   define HEPHAISTOS_API HEPHAISTOS_API_IMPORT
#endif

//DEBUG macro
#if !defined(NDEBUG)
#   define HEPHAISTOS_DEBUG
#endif
