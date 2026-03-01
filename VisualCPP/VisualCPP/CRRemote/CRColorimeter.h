//////////////////////////////////////////////////////////
// Copyright ©2015 Colorimetry Research, Inc. All Rights Reserved Worldwide. 	
// Version 1.26
//
// License: 
// This code is provided as a demonstration of the Remote Communication Software Development Kit.
// 
// This software is provided "as is" with no warranties of any kind.
// 
//////////////////////////////////////////////////////////

#pragma once

#ifndef __CCRCOLORIMETER_H__
#define __CCRCOLORIMETER_H__


#define __USE_REGISTERED_WINDOWS_MESSAGES

#ifndef CCRCOLORIMETER_EXT_CLASS
#define CCRCOLORIMETER_EXT_CLASS
#endif


#include "Channel.h"
#include "INotifier.h"

#include "..\SDK\flickerapi.h"
#include "..\SDK\responsetimeapi.h"
#include "..\SDK\colorscienceapi.h"

struct CRConfiguration;
struct CRSetup;
struct CRReading;

#ifndef __USE_REGISTERED_WINDOWS_MESSAGES
#define UWM_MEASUREMENT_CHANGED		WM_USER+1	
#define UWM_MEASUREMENT_DATA_CHANGED		WM_USER+2	
#define UWM_DATA_ERROR		WM_USER+3	
#define UWM_DATA_IGNORED		WM_USER+4	
#define UWM_DATA_SENT		WM_USER+5	
#define UWM_DATA_RECEIVED		WM_USER+6	
#define UWM_DATA_DEBUG		WM_USER+7	

#else
static UINT UWM_MEASUREMENT_CHANGED = ::RegisterWindowMessage(_T("UWM_MEASUREMENT_CHANGED-{7355AE1C-0C52-4924-9FAB-6E2CE71265A9}"));
static UINT UWM_MEASUREMENT_DATA_CHANGED = ::RegisterWindowMessage(_T("UWM_MEASUREMENT_DATA_CHANGED-{7355AE1C-0C52-4924-9FAB-6E2CE71265A9}"));
static UINT UWM_DATA_ERROR = ::RegisterWindowMessage(_T("UWM_DATA_ERROR-{7355AE1C-0C52-4924-9FAB-6E2CE71265A9}"));
static UINT UWM_DATA_IGNORED = ::RegisterWindowMessage(_T("UWM_DATA_IGNORED-{7355AE1C-0C52-4924-9FAB-6E2CE71265A9}"));
static UINT UWM_DATA_SENT = ::RegisterWindowMessage(_T("UWM_DATA_SENT-{7355AE1C-0C52-4924-9FAB-6E2CE71265A9}"));
static UINT UWM_DATA_RECEIVED = ::RegisterWindowMessage(_T("UWM_DATA_RECEIVED-{7355AE1C-0C52-4924-9FAB-6E2CE71265A9}"));
static UINT UWM_DATA_DEBUG = ::RegisterWindowMessage(_T("UWM_DATA_DEBUG-{7355AE1C-0C52-4924-9FAB-6E2CE71265A9}"));
static UINT UWM_CONNECTED = ::RegisterWindowMessage(_T("UWM_CONNECTED-{7355AE1C-0C52-4924-9FAB-6E2CE71265A9}"));
static UINT UWM_DISCONNECTED = ::RegisterWindowMessage(_T("UWM_DISCONNECTED-{7355AE1C-0C52-4924-9FAB-6E2CE71265A9}"));
#endif

static float __CR_FILTER_FREQUENCY_MAX = 800;
static float __CR_FILTER_FREQUENCY_MIN = 10;

static float __CR_FILTER_BANDWIDTH_MAX = 800;
static float __CR_FILTER_BANDWIDTH_MIN = 10;

static int __CR_FILTER_ORDER_MAX = 50;
static int __CR_FILTER_ORDER_MIN = 1;

static float __CR_MAX_SEARCH_FREQUENCY_MAX = 800;
static float __CR_MAX_SEARCH_FREQUENCY_DEFAULT = 120;
static float __CR_MAX_SEARCH_FREQUENCY_MIN = 10;

static int __CR_FILTER_MOVING_AVERAGE_MAX = 20;
static int __CR_FILTER_MOVING_AVERAGE_DEFAULT = 5;
static int __CR_FILTER_MOVING_AVERAGE_MIN = 1;

static float __CR_STEPZONE_MAX = 1.0F;
static float __CR_STEPZONE_MIN = 0.0F;

static float __CR_STEPZONE_LO_DEFAULT = 0.1F;
static float __CR_STEPZONE_HI_DEFAULT = 0.9F;

static float __CR_CLIPPING_MAX = 1.0F;
static float __CR_CLIPPING_MIN = 0.0F;

static float __CR_CLIPPING_LO_DEFAULT = 0.1F;
static float __CR_CLIPPING_HI_DEFAULT = 0.9F;

static float __CR_NOISELEVEL_MAX = 0.9F;
static float __CR_NOISELEVEL_MIN = 0.0F;

static float __CR_NOISELEVEL_DEFAULT = 0.05F;

static float __CR_DOMINANTWAVELENGTH_WHITEX = 0.3127F;
static float __CR_DOMINANTWAVELENGTH_WHITEY = 0.3290F;

static CString CStringFormat(LPCTSTR format, ...)
{
	va_list		ap;
	CString		str;

	va_start(ap, format);
	str.FormatV(format, ap);
	va_end(ap);		
	return str;
}

class CStringQueue : public CStringArray
{ 
public:
    // Go to the end of the line
    void Push( CString& newValue)
        { Add( newValue); }        // End of the queue

	// Get first element in line
	CString Pop()
	{ 
		CString value;
		if(!IsEmpty())
		{
			value = GetAt(0);
			RemoveAt(0); 
		}
		return value;
	}
	CString GetHead()
	{
		if(!IsEmpty())
		{
			return GetAt(0);
		}
		return CString();
	}
}; 

class CCRCOLORIMETER_EXT_CLASS CCRColorimeter
{
public:
	enum CRObserver
	{
		Observer_2Degree,
		Observer_10Degree,
	};

	CCRColorimeter(CWnd* owner = NULL);
	virtual ~CCRColorimeter(void);

	CWnd* Owner();
	void SetOwner(CWnd* owner);

	CChannel* Channel();
	void SetChannel(CChannel& channel);

	BOOL IsConnected() const;
	BOOL Connect();
	BOOL Disconnect();

	BOOL Capture();

	CString Firmware() const;
	CString ID() const;
	CString Model() const;

	int AccessoryCount() const;
	CString AccessoryName(int index) const;
	int AccessoryID(int index) const;
	int Accessory() const;
	void SetAccessory(int value);
	CString AccessoryType(int index) const;

    int MaxFilters() const;
    int FilterCount() const;
    CString FilterName(int Index) const;
	int FilterID(int Index) const;
    int FilterType(int Index) const;
    void ClearFilters();
    int Filter(int Index) const;
	void SetFilter(int Index, int value);

    int ApertureCount() const;
    CString ApertureName(int Index) const;
    int ApertureID(int Index) const;
    int Aperture() const;
    void SetAperture(int value);

    int ModeCount() const;
    int Mode() const;
    void SetMode(int value);
    int ModeID(int Index) const;
	CString ModeName(int Index) const;


    int ExposureModeCount() const;
    int ExposureMode() const;
	void SetExposureMode(int value);
    int ExposureModeID(int Index) const;
    CString ExposureModeName(int Index) const;

    int RangeModeCount() const;
    int RangeModeID(int Index) const; 
    int RangeMode() const;  
    void SetRangeMode(int value); 
    CString RangeModeName(int Index) const; 

    int RangeCount() const; 
    int RangeID(int Index) const; 
    int Range() const; 
	void SetRange(int value);
    CString RangeName(int Index) const; 

    int SpeedCount() const; 
    int SpeedID(int Index) const; 
    int Speed() const; 
	void SetSpeed(int value);
    CString SpeedName(int Index) const; 

    int SyncModeCount() const; 
    int SyncModeID(int Index) const; 
    int SyncMode() const; 
	void SetSyncMode(int value);
    CString SyncModeName(int Index) const; 

    float MinExposure() const; 
    float MaxExposure() const; 
    float Exposure() const; 
	void SetExposure(float value);
    float MaxAutoExposure() const; 
	void SetMaxAutoExposure(float value);

    float MinSyncFreq() const; 
    float MaxSyncFreq() const; 
    float SyncFreq() const; 
	void SetSyncFreq(float value);

    int MinExposureX() const; 
    int MaxExposureX() const; 
    int ExposureX() const; 
	void SetExposureX(int value);

    // This funtion is obsolete. Use UserCalibModeCount instead.
    int MatrixModeCount() const; 
    // This funtion is obsolete. Use UserCalibModeID instead.
    int MatrixModeID(int Index) const; 
    // This funtion is obsolete. Use UserCalibMode instead.
    int MatrixMode() const; 
    // This funtion is obsolete. Use SetUserCalibMode instead.
	void SetMatrixMode(int value);
    // This function is obsolete. Use UserCalibModeName instead.
    CString MatrixModeName(int Index) const; 


    int UserCalibModeCount() const; 
    int UserCalibModeID(int Index) const; 
    int UserCalibMode() const;
	void SetUserCalibMode(int value);
    CString UserCalibModeName(int Index) const; 

    int MatrixCount(int accessory) const; 
    int MatrixID(int accessory, int Index) const; 
    int Matrix(int accessory) const; 
    void SetMatrix(int accessory, int value); 
    CString MatrixName(int accessory, int Index) const; 

    int MatchCount() const; 
    int MatchID(int Index) const; 
    int Match() const; 
    void SetMatch(int value); 
    CString MatchName(int Index) const; 

	
    float MinSamplingRate() const; 
    float MaxSamplingRate() const; 
    float SamplingRate() const; 
	void SetSamplingRate(float value);

	// Flicker Settings
	int FlickerFilterType() const;
	void SetFlickerFilterType(int value);
	CString FlickerFilterTypeName() const;

	int FlickerFilterFamily() const;
	void SetFlickerFilterFamily(int value);

	int FlickerFilterOrder() const;
	void SetFlickerFilterOrder(int value);

	double FlickerFilterFrequency() const;
	void SetFlickerFilterFrequency(double value);

	double FlickerFilterBandwidth() const;
	void SetFlickerFilterBandwidth(double value);

	double FlickerMaxSearchFrequency() const;
	void SetFlickerMaxSearchFrequency(double value);
	
	// Response Time Settings
	
	int ResponseTimeFilterType() const;
	void SetResponseTimeFilterType(int value);
	CString ResponseTimeFilterTypeName() const;
	
	int ResponseTimeMode() const;
	void SetResponseTimeMode(int value);
	
	int ResponseTimeAverage() const;
	void SetResponseTimeAverage(int value);
		
	BOOL ResponseTimeClippingEnabled() const;
	void SetResponseTimeClippingEnabled(BOOL enabled);
		
	float ResponseTimeClippingLowerLimit() const;
	void SetResponseTimeClippingLowerLimit(float value);
		
	float ResponseTimeClippingUpperLimit() const;
	void SetResponseTimeClippingUpperLimit(float value);
	
	float ResponseTimeNoiseLevel() const;
	void SetResponseTimeNoiseLevel(float value);
	
	float ResponseTimeStepResponseZoneLowerLimit() const;
	void SetResponseTimeStepResponseZoneLowerLimit(float value);

	float ResponseTimeStepResponseZoneUpperLimit() const;
	void SetResponseTimeStepResponseZoneUpperLimit(float value);

	// CMF
	int MaxCMF() const;
	int CMF() const; 
	void SetCMF(int value);

    CString Reading(CString DataType) const; 
    CArray<double,double>& Spectrum() const;
    CArray<double,double>& Temporal() const;

    BOOL UploadSetup();

private:
	// Member variables
	CWnd* m_pOwner;
    CChannel* m_channel; 
    CRConfiguration* m_configuration;
    CRSetup* m_setup;
    CRSetup* m_setupModified;
    CRReading* m_reading;
    CStringQueue m_commandQ; 
    CStringQueue m_responseQ ;
    CStringQueue m_spentCommandQ;
    CStringQueue m_spentResponseQ;
    CString m_buffer;
    CMutex m_mutex;

	
    //Flicker variables
    cs_flicker_filter_t m_flickerFilter;
    cs_responsetime_settings_t m_responseTimeSettings;
    double m_flickerMaxSearchFrequency;

	void InitConfiguration();
	void InitSetup();
	void InitReading();
	void InitFlicker();
	void InitResponseTime();

	
	float VersionNumber();

	BOOL DownloadVersion();
	BOOL DownloadConfiguration();

	int AccessoryIDFromName(CString name) const;
	int AccessoryIndexFromID(int ID) const;
    CString AccessoryTypeFromName(CString name) const;


	int FilterIDFromName(CString Name) const;
	int FilterIndexFromID(int ID) const; 

	int ApertureIndexFromID(int ID) const; 
	int ApertureIDFromName(CString Name) const; 

	int ModeIDFromName(CString Name) const; 
	int ModeIndexFromID(int ID) const; 

	int ExposureModeIDFromName(CString Name) const; 
	int ExposureModeIndexFromID(int ID) const; 

	int RangeModeIDFromName(CString Name) const; 
	int RangeModeIndexFromID(int ID) const; 

	int RangeIDFromName(CString Name) const;
	int RangeIndexFromID(int ID) const; 

	int SpeedIDFromName(CString Name) const; 
	int SpeedIndexFromID(int ID) const; 

	int SyncModeIDFromName(CString Name) const; 
	int SyncModeIndexFromID(int ID) const; 

	// This property is obsolete. Use UserCalibModeIDFromName instead.
	int MatrixModeIDFromName(CString Name) const; 

	 // This property is obsolete. Use UserCalibModeIDFromName instead.
    int MatrixModeIndexFromID(int ID) const; 

	int UserCalibModeIDFromName(CString Name) const; 

	int UserCalibModeIndexFromID(int ID) const; 

	int MatrixIDFromName(int accessory, CString Name) const; 

	int MatrixIndexFromID(int accessory, int ID) const; 
	int MatchIDFromName(CString Name) const; 

	int MatchIndexFromID(int ID) const; 
	BOOL DownloadSetup(); 

	void OnCommandCompleted(CString Command);

	BOOL DownloadReading(); 

	BOOL SendCommand(CString command, int Timeout  = 1000); 
	         
	void DataReceivedHandler(const CString& buffer);
	void DataTransmittedHandler(const CString& buffer);
	void ProcessBuffer();
	void FlushBuffer();
	void ProcessResponses();
	
	void Split(CStringArray& saItems, CString sFrom, CString sToken)
	{
		int i = 0;
		for(CString sItem = sFrom.Tokenize(sToken,i); i >= 0; sItem = sFrom.Tokenize(sToken,i))
		{
			saItems.Add(sItem);
		}

	};

	private:
	class DataReceivedNotifier : public INotifier
	{
		DataReceivedNotifier(){}
		void OnNotifcation(NotificationType type, const CString& buffer);
		CCRColorimeter* m_meter;
		
		friend class CCRColorimeter;
	} m_dataReceivedNotifier;

};

#endif //__CCRCOLORIMETER_H__