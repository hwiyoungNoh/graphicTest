/*
 * Copyright ©2019 Colorimetry Research, Inc. <engineering@colorimetryresearch.com>
 *
 */
#ifndef COLORSCIENCEAPI_H
#define COLORSCIENCEAPI_H

#ifndef _MSC_VER
#include <stdint.h>
#include <sys/time.h>
#else
#if _MSC_VER < 1600
#include "ms_stdint.h"
#else
//#include <stdint.h> //including stdint causes macro redefintion warnings
#include "ms_stdint.h"
#endif
#include <time.h>
#endif

#include "colorscienceapi_global.h"
//#include "colorscienceapi_version.h"

COLORSCIENCEAPI_BEGIN_DECLS

// Dominant wavelength

//typedef struct _dominantwavelength dominantwavelength_t;

typedef struct {
    // input
    double xWhite;
    double yWhite;
    double x;
    double y;

    // result
    double dominantWavelength;
    double purity;
    bool isComplimentary;

} cs_dominantwavelength_data_t;

COLORSCIENCEAPI_EXPORT(int32_t) cs_dominantwavelength(cs_dominantwavelength_data_t *dominantwavelength_data);

COLORSCIENCEAPI_END_DECLS

#endif // COLORSCIENCEAPI_H
