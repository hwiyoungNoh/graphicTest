/*
 * Copyright ©2019 Colorimetry Research, Inc. <engineering@colorimetryresearch.com>
 *
 */

#ifndef COLORSCIENCEAPI_RESPONSETIMEAPI_H
#define COLORSCIENCEAPI_RESPONSETIMEAPI_H

#ifndef _MSC_VER
#include <stdint.h>
#include <sys/time.h>
#else
#include "ms_stdint.h"
#include <time.h>
#endif

#include "colorscienceapi_global.h"
//#include "colorscienceapi_version.h"


COLORSCIENCEAPI_BEGIN_DECLS

typedef struct _responsetime responsetime_t;

typedef struct {
    double sampling_rate;
    uint32_t samples;

    double minimum;
    double maximum;
    double contrast;

    double valley1;
    double peak;
    double valley2;

    double rise_time;
    double fall_time;
    double response_time;

} cs_responsetime_data_t;

typedef struct {
    uint32_t mode;  // responsetime_mode

    // manual settings
    uint8_t filter_type;    // responsetime_filter_type

    // moving window average settings
    uint8_t average;

    // clipping settings
    uint8_t clipping_enabled;
    double clipping_lo; // %
    double clipping_hi; // %

    //peak/valley filter
    double noiselevel;  // %

    // step response zone settings
    double setupresponsezone_lo; // %
    double setupresponsezone_hi; // %

} cs_responsetime_settings_t;


typedef enum
{
    RT_FILTER_TYPE_NONE                = 0,
    RT_FILTER_TYPE_MOVINGWINDOWAVERAGE = 1
} responsetime_filter_type;


typedef enum
{
    RT_MODE_AUTO    = 0,
    RT_MODE_MANUAL  = 1
} responsetime_mode;


/* create/destroy */
COLORSCIENCEAPI_EXPORT(responsetime_t*) cs_responsetime_create(double sampling_rate, double *data, uint32_t count);
COLORSCIENCEAPI_EXPORT(void) cs_responsetime_free(responsetime_t *ctx);

/* response time parameters */
COLORSCIENCEAPI_EXPORT(int32_t) cs_responsetime_update(responsetime_t *ctx, cs_responsetime_settings_t *responsetime_settings);
COLORSCIENCEAPI_EXPORT(int32_t) cs_responsetime_peaks(responsetime_t *ctx, uint8_t *peaks);

// peak = 0 returns min, max, contrast values for the whole signal. peak > 0 and < cs_responsetime_peaks return data for the single waveform.
COLORSCIENCEAPI_EXPORT(int32_t) cs_responsetime_data(responsetime_t *ctx, uint8_t peak, cs_responsetime_data_t *responsetime_data);


COLORSCIENCEAPI_EXPORT(const char *)cs_responsetime_strerror(uint32_t errnum);

COLORSCIENCEAPI_END_DECLS

#endif // !COLORSCIENCEAPI_RESPONSETIMEAPI_H
