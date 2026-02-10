#ifndef COLORSCIENCEAPI_GLOBAL_H
#define COLORSCIENCEAPI_GLOBAL_H


#if defined(COLORSCIENCEAPI_LIBRARY)
#ifdef _MSC_VER
#  define COLORSCIENCEAPI_EXPORT(T) __declspec (dllexport) T
#else // !_MSC_VER
#  define COLORSCIENCEAPI_EXPORT(T) T
#endif
#else // !defined(COLORSCIENCEAPI_LIBRARY)
#ifdef _MSC_VER
#  define COLORSCIENCEAPI_EXPORT(T) __declspec (dllimport) T
#else // !_MSC_VER
#  define COLORSCIENCEAPI_EXPORT(T) T
#endif
#endif // defined(COLORSCIENCEAPI_LIBRARY)

#ifdef  __cplusplus
# define COLORSCIENCEAPI_BEGIN_DECLS  extern "C" {
# define COLORSCIENCEAPI_END_DECLS    }
#else
# define COLORSCIENCEAPI_BEGIN_DECLS
# define COLORSCIENCEAPI_END_DECLS
#endif


#endif // COLORSCIENCEAPI_GLOBAL_H
