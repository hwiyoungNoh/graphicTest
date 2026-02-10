/*
 * Copyright ©2019 Colorimetry Research, Inc. <engineering@colorimetryresearch.com>
 *
 */

#ifndef COLORSCIENCEAPI_FLICKERAPI_H
#define COLORSCIENCEAPI_FLICKERAPI_H

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


typedef struct _flicker flicker_t;


typedef struct {
    double sampling_rate;
    uint32_t samples;

    double percent_flicker;
    double flicker_index;
    double flicker_modulation_amplitude;

    double maximum_flicker_search_frequency;
    double flicker_frequency;
    double flicker_level;

    double minimum_frequency_index;
    double maximum_frequency_index;
    double minimum_frequency;
    double maximum_frequency;

} cs_flicker_data_t;

typedef enum
{
    FILTER_FAMILY_NONE          = 0,
    FILTER_FAMILY_BUTTERWORTH   = 1
} flicker_filter_family;


typedef enum
{
    FILTER_TYPE_NONE          = 0,
    FILTER_TYPE_LOWPASS       = 1,
    FILTER_TYPE_HIGHPASS      = 2,
    FILTER_TYPE_BANDPASS      = 3,
    FILTER_TYPE_BANDSTOP      = 4
} flicker_filter_type;


typedef struct {
    uint8_t filter_type;
    uint8_t filter_family; //reserved for future use
    uint8_t order;
    double frequency;
    double bandwidth;

} cs_flicker_filter_t;


/* filter */
COLORSCIENCEAPI_EXPORT(int32_t) cs_flicker_filter(cs_flicker_filter_t* filter, double sampling_rate, double *data, uint32_t count, double *filtered_data);

/* create/destroy */
COLORSCIENCEAPI_EXPORT(flicker_t*) cs_flicker_create(double sampling_rate, double *data, uint32_t count, double maximumSearchFrequency);
COLORSCIENCEAPI_EXPORT(void) cs_flicker_free(flicker_t *ctx);

/* flicker parameters */
COLORSCIENCEAPI_EXPORT(int32_t) cs_flicker_fft(flicker_t *ctx, double *fft_data, uint32_t *count);
COLORSCIENCEAPI_EXPORT(int32_t) cs_flicker_data_ex(cs_flicker_data_t *&flicker_data, double sampling_rate, double *data, uint32_t count, double maximumSearchFrequency);
COLORSCIENCEAPI_EXPORT(int32_t) cs_flicker_data(flicker_t *ctx, cs_flicker_data_t *flicker_data);


COLORSCIENCEAPI_EXPORT(const char *)cs_flicker_strerror(uint32_t errnum);

COLORSCIENCEAPI_END_DECLS

#endif // !COLORSCIENCEAPI_FLICKERAPI_H
