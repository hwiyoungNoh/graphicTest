//////////////////////////////////////////////////////////
// Copyright ©2018 Colorimetry Research, Inc. All Rights Reserved Worldwide.
// Version 1.26
//
// License: 
// This code is provided as a demonstration of the Remote Communication Software Development Kit.
// 
// This software is provided "as is" with no warranties of any kind.
// 
//////////////////////////////////////////////////////////

#include "StdAfx.h"
#include "CRColorimeter.h"

#define NEW_LINE _T("\r\n")

#define RESPONSE_SEPARATOR _T(":")
#define RESULT_SEPARATOR _T("),")
#define RESULT_SPACE _T(" ")
#define RESPONSE_OK _T("OK")
#define RESPONSE_ERROR _T("ER")
#define RESPONSE_MSECS _T("msec")
#define RESPONSE_HZ _T("Hz")

struct CRMode
{
	CRMode()
	{
		ID = 0;
	}
    int ID;
    CString Name;
};


struct CRMatrix
{
	CRMatrix()
	{
		ID = 0;
	}

	CRMatrix(const CRMatrix& other)
	{
		ID = other.ID;
		Name = other.Name;
		Calibration.Copy(other.Calibration);
	}

	CRMatrix& operator=(const CRMatrix &other) 
	{
		ID = other.ID;
		Name = other.Name;
		Calibration.Copy(other.Calibration);
		return *this;
	}

    CString Name;
    int ID;
    CArray<float, float> Calibration;
};


struct CRMatch
{		
	CRMatch()
	{
		ID = 0;
	}

	CRMatch(const CRMatch& other)
	{
		ID = other.ID;
		Name = other.Name;
		Calibration.Copy(other.Calibration);
	}

	CRMatch& operator=(const CRMatch &other) 
	{
		ID = other.ID;
		Name = other.Name;
		Calibration.Copy(other.Calibration);
		return *this;
	}

    CString Name;
    int ID;
    CArray<float, float> Calibration;
};

struct CRAccessory
{
	CRAccessory()
	{
		ID = 0;
	}
	
	CRAccessory(const CRAccessory& other)
	{
		ID = other.ID;
		Name = other.Name;
		Type = other.Type;
		Matrices.Copy(other.Matrices);
	}

	CRAccessory& operator=(const CRAccessory &other) 
	{
		ID = other.ID;
		Name = other.Name;
		Type = other.Type;
		Matrices.Copy(other.Matrices);
		return *this;
	}

	int ID;
	CString Name;
	CString Type;
	CArray<CRMatrix> Matrices;
};

struct CRAperture
{
	CRAperture()
	{
		ID = 0;
	}
	int ID;
	CString Name;
};

struct CRFilter
{
	CRFilter()
	{
		ID = 0;
	}
	int ID;
	CString Name;
	CString Type;
};

struct CRExposureMode
{
	CRExposureMode()
	{
		ID = 0;
	}
	int ID;
	CString Name;
};

struct CRRangeMode
{
	CRRangeMode()
	{
		ID = 0;
	}
    int ID;
    CString Name;
};

struct CRRange
{
	CRRange()
	{
		ID = 0;
	}
	int  ID;
	CString Name;
};

struct CRSpeed
{
	CRSpeed()
	{
		ID = 0;
	}
    int ID;
    CString Name;
};

struct CRSyncMode
{
	CRSyncMode()
	{
		ID = 0;
	}
	int ID;
	CString Name;
};

struct CRMatrixMode
{
	CRMatrixMode()
	{
		ID = 0;
	}
    int ID;
    CString Name;
};

struct CRUserCalibMode
{
	CRUserCalibMode()
	{
		ID = 0;
	}
    int ID;
    CString Name;
};


struct CRConfiguration
{
	CRConfiguration()
	{
		InstrumentType = 0;
		ModesCount = 0;
		AccessoryCount = 0;
		FilterCount = 0;
		ApertureCount = 0;
		ExposureModesCount = 0;
		RangeModesCount = 0;
		RangesCount = 0;
		SpeedsCount = 0;
		SyncModesCount = 0;
		MinExposure = 0;
		MaxExposure = 0;
		MinSyncFreq = 0;
		MaxSyncFreq = 0;
		MinExposureX = 0;
		MaxExposureX = 0;
		MatchCount = 0;
		MinSamplingRate = 0;
		MaxSamplingRate = 0;
	}
    CString ID;
    CString Model;
    int InstrumentType;
    int ModesCount;// Since 1.16
    CArray<CRMode> Modes; // Since 1.16

    int AccessoryCount;
    CArray<CRAccessory> Accessories;

    int FilterCount;
    CArray<CRFilter> Filters;

    int ApertureCount;
    CArray<CRAperture> Apertures;

    int ExposureModesCount;
    CArray<CRExposureMode> ExposureModes;

    int RangeModesCount;
    CArray<CRRangeMode> RangeModes;

    CArray<CRRange> Ranges;
    int RangesCount;

    CArray<CRSpeed> Speeds;
    int SpeedsCount;

    CArray<CRSyncMode> SyncModes;
    int SyncModesCount;

    CString Firmware;

    CArray<CRMatrixMode> MatrixModes; // Deprecated as of Firmware 1.16
    CArray<CRUserCalibMode> UserCalibModes; // Since 1.16

    float MinExposure;
    float MaxExposure;

    float MinSyncFreq;
    float MaxSyncFreq;

    int MinExposureX;
    int MaxExposureX;

    CArray<CRMatch> MatchSet; // Since 1.16
    int MatchCount; // Since 1.16

    float MinSamplingRate; // Since 1.19
    float MaxSamplingRate; // Since 1.19

};

struct CRSetup
{
    int ModeID;
    int AccessoryID;
    int Filter1ID;
    int Filter2ID;
    int Filter3ID;
    int ApertureID;
    int RangeModeID;
    int RangeID;
    int SpeedID;
    int ExposureModeID;
    float Exposure;
    float MaxAutoExposure; // Since 1.26
    int SyncModeID;
    float SyncFreq;
    int ExposureX;
    int MatrixModeID;
    int UserCalibModeID;
    int MatrixID;
    int MatchID;
    float SamplingRate; // Since 1.19
    int CMF; // Since 1.26
};



struct CRWarning
{
	CRWarning()
	{
		Code = 0;
	}
	int Code;
	CString Description;
};

struct CRCIE
{
    CString X;
    CString Y;
    CString Z;
    CString XYZ;
    CString xy;
    CString uv;
    CString upvp;
    CString CCT;
};


struct CRSpectrum  // Since 1.17
{
	CRSpectrum()
	{
		StartingWavelength = 0;
		EndingWavelength = 0;
		Delta = 0;
	}
	
	CRSpectrum(const CRSpectrum& other)
	{
		StartingWavelength = other.StartingWavelength;
		EndingWavelength = other.EndingWavelength;
		Delta = other.Delta;
		Data.Copy(other.Data);
	}

	CRSpectrum& operator=(const CRSpectrum &other) 
	{
		StartingWavelength = other.StartingWavelength;
		EndingWavelength = other.EndingWavelength;
		Delta = other.Delta;
		Data.Copy(other.Data);
		return *this;
	}

    float StartingWavelength;
    float EndingWavelength;
    float Delta;
    CArray<double,double> Data;
};


struct CRTemporal  // Since 1.17
{
	CRTemporal()
	{
		SamplingRate = 0;
	}
	
	CRTemporal(const CRTemporal& other)
	{
		SamplingRate = other.SamplingRate;
		Data.Copy(other.Data);
	}

	CRTemporal& operator=(const CRTemporal &other) 
	{
		SamplingRate = other.SamplingRate;
		Data.Copy(other.Data);
		return *this;
	}

    float SamplingRate;
    CArray<double,double> Data;
};


struct CRReading 
{
	CString ID;
	CString Model;
	CString Time;
	CString Mode;
	CString Accessory;
	CString Filter;
	CString Aperture;
	CString ExposureMode;
	CString Exposure;
	CString MaxAutoExposure; // Since 1.26
	CString RangeMode;
	CString Range;
	CString Speed;
	CString SyncMode;
	CString SyncFreq;
	CString ExposureX;
	CString MatrixMode;
	CString UserCalibMode;
	CString MatrixID;
	CString MatchID;
	CRCIE CIE[2];
	CString Yv;
	CString Radiometric;
	CString CMF; // Since 1.26
	CArray<CRWarning> Warnings;
	CString AllWarnings;
	CRSpectrum Spectrum;
	CRTemporal Temporal;
};

CCRColorimeter::CCRColorimeter(CWnd* owner)
{
	m_dataReceivedNotifier.m_meter = this;
	m_pOwner = owner;
	m_channel = NULL;

	m_configuration = NULL;
    m_setup = NULL;
    m_setupModified = NULL;
    m_reading = NULL;

	InitConfiguration();
	InitSetup();
	InitReading();
	InitFlicker();
	InitResponseTime();
}

CCRColorimeter::~CCRColorimeter(void)
{
	// do not delete the channnel as it is not owned by this object.
	m_channel = NULL;
	
	if(m_configuration)
		delete m_configuration;
	m_configuration = NULL;

	if(m_setup)
		delete m_setup;
	m_setup = NULL;

	if(m_setupModified)
		delete m_setupModified;
	m_setupModified = NULL;

	if(m_reading)
		delete m_reading;
	m_reading = NULL;
}

CWnd* CCRColorimeter::Owner()
{
	return m_pOwner;
}

void CCRColorimeter::SetOwner(CWnd* owner)
{
	m_pOwner = owner;
}

CChannel* CCRColorimeter::Channel()
{
	return m_channel;
}

void CCRColorimeter::SetChannel(CChannel& channel)
{
	m_channel = &channel;
	m_channel->SetNotifier(&m_dataReceivedNotifier);
}

BOOL CCRColorimeter::IsConnected() const
{
	if(m_channel)
		return m_channel->IsOpen();
	return FALSE;
}

BOOL CCRColorimeter::Connect()
{
	m_commandQ.RemoveAll();
	m_responseQ.RemoveAll();
	m_spentCommandQ.RemoveAll();
	m_spentResponseQ.RemoveAll();

	BOOL result = FALSE;
	if(!m_channel)
		return FALSE;

	if(!m_channel->IsOpen())
	{	
		m_channel->Open();		
	}
	result = m_channel->IsOpen();

	if(result)
	{
		::SendMessage(m_pOwner->m_hWnd, UWM_CONNECTED, (WPARAM)0, (LPARAM) this);
		result = DownloadVersion();
	}

	return result;
}

BOOL CCRColorimeter::Disconnect()
{
	BOOL result = FALSE;

	if(!m_channel)
		return FALSE;

	if(m_channel->IsOpen())
	{
		m_channel->Close();		
		::SendMessage(m_pOwner->m_hWnd, UWM_DISCONNECTED, (WPARAM)0, (LPARAM) this);
	}
	return TRUE;
}

BOOL CCRColorimeter::Capture()
{
	UploadSetup();
	SendCommand(_T("M"));
	return TRUE;
}

void CCRColorimeter::InitConfiguration()
{
	if(m_configuration)
		delete m_configuration;
	m_configuration = new CRConfiguration();
}

void CCRColorimeter::InitSetup()
{
	if(m_setup)
		delete m_setup;
	m_setup = new CRSetup();
	
	if(m_setupModified)
		delete m_setupModified;
	m_setupModified = new CRSetup();
}

void CCRColorimeter::InitReading()
{
	if(m_reading)
		delete m_reading;
	m_reading = new CRReading();
}

void CCRColorimeter::InitFlicker()
{
	m_flickerFilter.filter_type = FILTER_TYPE_NONE;
	m_flickerFilter.filter_family = 0;
	m_flickerFilter.order = __CR_FILTER_ORDER_MIN;
	m_flickerFilter.frequency = __CR_FILTER_FREQUENCY_MAX;
	m_flickerFilter.bandwidth = __CR_FILTER_BANDWIDTH_MAX;
	m_flickerMaxSearchFrequency = __CR_MAX_SEARCH_FREQUENCY_DEFAULT;
}

    // clipping settings
    uint8_t clipping_enabled;
    double clipping_lo; // %
    double clipping_hi; // %

    //peak/valley filter
    double noiselevel;  // %

    // step response zone settings
    double setupresponsezone_lo; // %
    double setupresponsezone_hi; // %

void CCRColorimeter::InitResponseTime()
{
	m_responseTimeSettings.mode = RT_MODE_AUTO;
	m_responseTimeSettings.filter_type = RT_FILTER_TYPE_NONE;
	m_responseTimeSettings.average = __CR_FILTER_MOVING_AVERAGE_DEFAULT;
	m_responseTimeSettings.clipping_enabled = FALSE;
	m_responseTimeSettings.clipping_lo = __CR_CLIPPING_LO_DEFAULT;
	m_responseTimeSettings.clipping_hi = __CR_CLIPPING_HI_DEFAULT;
	
	m_responseTimeSettings.noiselevel = __CR_NOISELEVEL_DEFAULT;

	m_responseTimeSettings.setupresponsezone_lo = __CR_STEPZONE_LO_DEFAULT;
	m_responseTimeSettings.setupresponsezone_hi = __CR_STEPZONE_HI_DEFAULT;

}

BOOL CCRColorimeter::DownloadVersion()
{
	SendCommand(_T("RC Firmware"));
	return TRUE;
}

float CCRColorimeter::VersionNumber()
{
    double version = strtod(CStringA(m_configuration->Firmware).GetString(), NULL);

	return (float)version;
}

BOOL CCRColorimeter::DownloadConfiguration()
{	
	SendCommand(_T("RC ID"));
	SendCommand(_T("RC Model"));
	float Version = VersionNumber();
	if (Version >= 1.17F)
		SendCommand(_T("RC InstrumentType"));


	SendCommand(_T("RC Accessory"));
	SendCommand(_T("RC Filter"));
	SendCommand(_T("RC Aperture"));
	SendCommand(_T("RC Mode"));
	SendCommand(_T("RC ExposureMode"));
	SendCommand(_T("RC RangeMode"));
	SendCommand(_T("RC Range"));
	SendCommand(_T("RC Speed"));
	SendCommand(_T("RC SyncMode"));
	SendCommand(_T("RC MatrixMode"));
	SendCommand(_T("RC UserCalibMode"));
	//SendCommand(_T("RC Matrix")); //This command is sent for each accessory ID when the RC Accessory returns
	//SendCommand(_T("RC MatrixCalibration")); // This command is sent for each accessory ID when the RC Accessory returns
	SendCommand(_T("RC Match"));
	SendCommand(_T("RC MinExposure"));
	SendCommand(_T("RC MaxExposure"));
	SendCommand(_T("RC MinSyncFreq"));
	SendCommand(_T("RC MaxSyncFreq"));
	SendCommand(_T("RC MinExposureX"));
	SendCommand(_T("RC MaxExposureX"));
	SendCommand(_T("RC MinSamplingRate"));
	SendCommand(_T("RC MaxSamplingRate"));

	return TRUE;
}

BOOL CCRColorimeter::DownloadSetup()
{
	SendCommand(_T("RS Accessory"));
    SendCommand(_T("RS Filter"));
    SendCommand(_T("RS Aperture"));
    SendCommand(_T("RS Mode"));
    SendCommand(_T("RS RangeMode"));
    SendCommand(_T("RS Range"));
    SendCommand(_T("RS Speed"));
    SendCommand(_T("RS ExposureMode"));
    SendCommand(_T("RS Exposure"));
    SendCommand(_T("RS MaxAutoExposure"));
    SendCommand(_T("RS SyncMode"));
    SendCommand(_T("RS SyncFreq"));
    SendCommand(_T("RS ExposureX"));
    SendCommand(_T("RS MatrixMode"));
    SendCommand(_T("RS UserCalibMode"));
    SendCommand(_T("RS Matrix"));
    SendCommand(_T("RS Match"));
    SendCommand(_T("RS SamplingRate"));
    SendCommand(_T("RS CMF"));

    return TRUE;
}

BOOL CCRColorimeter::DownloadReading()
{
	SendCommand(_T("RM ID"));
	SendCommand(_T("RM Model"));
	SendCommand(_T("RM Time"));
	SendCommand(_T("RM Accessory"));
	SendCommand(_T("RM Filter"));
	SendCommand(_T("RM Aperture"));
	SendCommand(_T("RM Mode"));
	SendCommand(_T("RM ExposureMode"));
	SendCommand(_T("RM Exposure"));
	SendCommand(_T("RM MaxAutoExposure"));
	SendCommand(_T("RM RangeMode"));
	SendCommand(_T("RM Range"));
	SendCommand(_T("RM Speed"));
	SendCommand(_T("RM SyncMode"));
	SendCommand(_T("RM SyncFreq"));
	SendCommand(_T("RM ExposureX"));
	//SendCommand(_T("RM MatrixMode"));
	SendCommand(_T("RM UserCalibMode"));
	SendCommand(_T("RM Matrix"));
	SendCommand(_T("RM Match"));
	SendCommand(_T("RM CMF"));
	SendCommand(_T("RM X"));
	SendCommand(_T("RM Y"));
	SendCommand(_T("RM Z"));
	SendCommand(_T("RM XYZ"));
	SendCommand(_T("RM xy"));
	SendCommand(_T("RM uv"));
	SendCommand(_T("RM upvp"));
	SendCommand(_T("RM CCT"));
	SendCommand(_T("RM X10"));
	SendCommand(_T("RM Y10"));
	SendCommand(_T("RM Z10"));
	SendCommand(_T("RM XYZ10"));
	SendCommand(_T("RM xy10"));
	SendCommand(_T("RM Warnings"));
	SendCommand(_T("RM Spectrum"));
	Sleep(150);
	SendCommand(_T("RM Radiometric"));
	SendCommand(_T("RM Yv"));
	Sleep(150);

	if (VersionNumber() >= 1.19F)
		SendCommand(_T("RM TemporalY"));
	else
		SendCommand(_T("RM Temporal"));
	return TRUE;
}


void CCRColorimeter::OnCommandCompleted(CString Command )
{
	if (Command == "RC Firmware") 
	{
		DownloadConfiguration();
		DownloadSetup();
	}
	else
	{
	}
}

CString CCRColorimeter::Firmware() const
{
	return m_configuration->Firmware;
}

CString CCRColorimeter::ID() const
{
	return m_configuration->ID;
}

CString CCRColorimeter::Model() const
{
	return m_configuration->Model;
}

int CCRColorimeter::AccessoryCount() const
{
	return m_configuration->AccessoryCount;
}

CString CCRColorimeter::AccessoryName(int index) const
{
	return m_configuration->Accessories[index].Name;
}

int CCRColorimeter:: AccessoryIDFromName(CString Name) const
{
	int id  = -1;
	if (Name == _T("None"))
	{
		return id;
	}
	for(int i = 0;i< m_configuration->Accessories.GetCount(); i++)
	{
		if (m_configuration->Accessories[i].Name == Name) 
		{
			id = m_configuration->Accessories[i].ID;
			break;
		}
	}
	return id;
}

int CCRColorimeter::AccessoryID(int index) const
{
	return m_configuration->Accessories[index].ID;
}

int CCRColorimeter::AccessoryIndexFromID(int ID) const
{
	for(int index=0; index < AccessoryCount(); index++)
	{
		if(m_configuration->Accessories[index].ID == ID) 
			return index;
	}
	return -1;
}

int CCRColorimeter::Accessory() const
{
	return AccessoryIndexFromID(m_setup->AccessoryID);
}

void CCRColorimeter::SetAccessory(int value)
{
	if (value >= 0 && value < AccessoryCount()) 
	{
		m_setupModified->AccessoryID = m_configuration->Accessories[value].ID;
	}
}

CString CCRColorimeter::AccessoryTypeFromName(CString Name) const
{
	CString Type = _T("");
	if (Name == _T("None")) 
	{
		return _T("NA");
	}
	for(int i=0; i < AccessoryCount(); i++)
	{
		if (m_configuration->Accessories[i].Name == Name) 
		{
			Type = m_configuration->Accessories[i].Type;
			break;
		}
	}
	return Type;
}

CString CCRColorimeter::AccessoryType(int index) const
{
	return m_configuration->Accessories[index].Type;
}

int CCRColorimeter::MaxFilters() const
{
	return 3;
}

int CCRColorimeter::FilterCount() const
{
	return (int)m_configuration->Filters.GetCount();
}

CString CCRColorimeter::FilterName(int Index) const
{
	return m_configuration->Filters[Index].Name;
}

int CCRColorimeter::FilterIDFromName(CString Name) const
{
	int Id  = -1;
	if (Name == _T("None")) 
	{
		return Id;
	}
	for(int i=0; i < FilterCount(); i++)
	{
		if (m_configuration->Filters[i].Name == Name) 
		{
			Id = m_configuration->Filters[i].ID;
		}
	}
	return Id;
}

int CCRColorimeter::FilterID(int Index) const
{
	return m_configuration->Filters[Index].ID;
}

int CCRColorimeter::FilterType(int Index) const
{
	return 0;
}

int CCRColorimeter::FilterIndexFromID(int ID) const
{
	for(int index=0; index<FilterCount(); index++)
	{
		if (m_configuration->Filters[index].ID == ID) return index;
	}
	return -1;
}

void CCRColorimeter::ClearFilters()
{
	m_setupModified->Filter1ID = -1;
	m_setupModified->Filter2ID = -1;
	m_setupModified->Filter3ID = -1;
}

int CCRColorimeter::Filter(int Index) const
{
	switch(Index)
	{
	case 0:
		return FilterIndexFromID(m_setupModified->Filter1ID);
		break;
	case 1:
		return FilterIndexFromID(m_setupModified->Filter2ID);
		break;
	case 2:
		return FilterIndexFromID(m_setupModified->Filter3ID);
		break;
	default:
		return -1;
		break;
	}
}

void CCRColorimeter::SetFilter(int Index, int value)
{
	if (value >= -1 && value < FilterCount()) 
	{
		switch(Index)
		{
		case 0:
			m_setupModified->Filter1ID = m_configuration->Filters[value].ID;
			break;
		case 1:
			m_setupModified->Filter2ID = m_configuration->Filters[value].ID;
			break;
		case 2:
			m_setupModified->Filter3ID = m_configuration->Filters[value].ID;
			break;
		default:
			break;
		}
	}
}

int CCRColorimeter::ApertureCount() const
{
	return (int)m_configuration->Apertures.GetCount();
}

CString CCRColorimeter::ApertureName(int Index) const
{
	return m_configuration->Apertures[Index].Name;
}

int CCRColorimeter::ApertureID(int Index) const
{
	return m_configuration->Apertures[Index].ID;
}

int CCRColorimeter::ApertureIndexFromID(int ID) const
{
	for(int index = 0; index<ApertureCount(); index++)
	{
		if (m_configuration->Apertures[index].ID == ID) return index;
	}
	return -1;
}

int CCRColorimeter::Aperture() const
{
	return ApertureIndexFromID(m_setupModified->ApertureID);
}

void CCRColorimeter::SetAperture(int value)
{
	if (value >= 0 && value < ApertureCount()) 
	{
		m_setupModified->ApertureID = m_configuration->Apertures[value].ID;
	}
}

int CCRColorimeter::ApertureIDFromName(CString Name) const
{
	int Id = -1;
	if (Name == _T("None")) 
	{
		return Id;
	}
	for(int i=0; i < ApertureCount(); i++)
	{
		if (m_configuration->Apertures[i].Name == Name) 
		{
			Id = m_configuration->Apertures[i].ID;
			break;
		}
	}
	return Id;
}

int CCRColorimeter::ModeCount() const
{
	return (int)m_configuration->Modes.GetCount();
}

int CCRColorimeter::ModeIDFromName(CString Name) const
{
	int Id = -1;
	for(int i=0; i < ModeCount(); i++)
	{
		if (m_configuration->Modes[i].Name == Name) 
		{
			Id = m_configuration->Modes[i].ID;
			break;
		}
	}
	return Id;
}

int CCRColorimeter::Mode() const
{
	return  ModeIndexFromID(m_setupModified->ModeID);
}

void CCRColorimeter::SetMode(int value)
{
	if (value >= 0 && value < ModeCount()) 
	{
		m_setupModified->ModeID = m_configuration->Modes[value].ID;
	}
}

int CCRColorimeter::ModeID(int Index) const
{
	return m_configuration->Modes[Index].ID;
}

int CCRColorimeter::ModeIndexFromID(int ID) const
{
	for(int index=0; index < ModeCount(); index++)
	{
		if (m_configuration->Modes[index].ID == ID) return index;
	}
	return -1;
}

CString CCRColorimeter::ModeName(int Index) const
{
	return m_configuration->Modes[Index].Name;
}

int CCRColorimeter::ExposureModeCount() const
{
	return (int)m_configuration->ExposureModes.GetCount();
}

int CCRColorimeter::ExposureModeIDFromName(CString Name) const
{
	int Id = -1;
	for(int i=0; i < ExposureModeCount(); i++)
	{
		if (m_configuration->ExposureModes[i].Name == Name) 
		{
			Id = m_configuration->ExposureModes[i].ID;
			break;
		}
	}
	return Id;
}

int CCRColorimeter::ExposureMode() const
{
	return ExposureModeIndexFromID(m_setupModified->ExposureModeID);
}

void CCRColorimeter::SetExposureMode(int value)
{
	if (value >= 0 && value < ExposureModeCount()) 
	{
		m_setupModified->ExposureModeID = m_configuration->ExposureModes[value].ID;
	}
}

int CCRColorimeter::ExposureModeID(int Index) const
{
	return m_configuration->ExposureModes[Index].ID;
}

int CCRColorimeter::ExposureModeIndexFromID(int ID) const
{
	for(int index = 0; index < ExposureModeCount(); index++)
	{
		if (m_configuration->ExposureModes[index].ID == ID) return index;
	}
	return -1;
}

CString CCRColorimeter::ExposureModeName(int Index) const
{
	return m_configuration->ExposureModes[Index].Name;
}

int CCRColorimeter::RangeModeCount() const
{
	return (int)m_configuration->RangeModes.GetCount();
}

int CCRColorimeter::RangeModeID(int Index) const
{
	return m_configuration->RangeModes[Index].ID;
}

int CCRColorimeter::RangeMode() const
{
	return RangeModeIndexFromID(m_setupModified->RangeModeID);
}

void CCRColorimeter::SetRangeMode(int value)
{
	if (value >= 0) 
	{
        m_setupModified->RangeModeID = m_configuration->RangeModes[value].ID;
	}
}

int CCRColorimeter::RangeModeIDFromName(CString Name) const
{
	int Id = -1;
	for(int i = 0;i< RangeModeCount(); i++)
	{
		if (m_configuration->RangeModes[i].Name == Name) 
		{
			Id = m_configuration->RangeModes[i].ID;
			break;
		}
	}
	return Id;
}

int CCRColorimeter::RangeModeIndexFromID(int ID) const
{
	for(int index=0; index < RangeModeCount(); index++)
	{
		if(m_configuration->RangeModes[index].ID == ID) return index;
	}
	return -1;
}

CString CCRColorimeter::RangeModeName(int Index) const
{
	return m_configuration->RangeModes[Index].Name;
}

int CCRColorimeter::RangeCount() const
{
	return (int)m_configuration->Ranges.GetCount();
}

int CCRColorimeter::RangeID(int Index) const
{
	return  m_configuration->Ranges[Index].ID;
}

int CCRColorimeter::Range() const
{
	return RangeIndexFromID(m_setupModified->RangeID);
}

void CCRColorimeter::SetRange(int value)
{ 
	if (value >= 0 && value < RangeCount()) 
	{
		m_setupModified->RangeID = m_configuration->Ranges[value].ID;
	}
}

int CCRColorimeter::RangeIDFromName(CString Name) const
{
	int Id = -1;
	for(int i=0; i < RangeCount(); i++)
	{
		if (m_configuration->Ranges[i].Name == Name) 
		{
			Id = m_configuration->Ranges[i].ID;
			break;
		}
	}
	return Id;
}

int CCRColorimeter::RangeIndexFromID(int ID)const
{
	for(int index=0; index < RangeCount(); index++)
	{
		if (m_configuration->Ranges[index].ID == ID) return index;
	}
	return -1;
}

CString CCRColorimeter::RangeName(int Index) const
{
	return m_configuration->Ranges[Index].Name;
}

int CCRColorimeter::SpeedCount() const
{
	return (int)m_configuration->Speeds.GetCount();
}

int CCRColorimeter::SpeedID(int Index) const
{
	return m_configuration->Speeds[Index].ID;
}

int CCRColorimeter::Speed() const
{
	return SpeedIndexFromID(m_setupModified->SpeedID);
}

void CCRColorimeter::SetSpeed(int value)
{ 
	if (value >= 0 && value < SpeedCount()) 
	{
		m_setupModified->SpeedID = m_configuration->Speeds[value].ID;
	}
}

int CCRColorimeter::SpeedIDFromName(CString Name) const
{
	int Id = -1;
	for(int i=0; i < SpeedCount(); i++)
	{
		if (m_configuration->Speeds[i].Name == Name) 
		{
			Id = m_configuration->Speeds[i].ID;
			break;
		}
	}
	return Id;
}

int CCRColorimeter::SpeedIndexFromID(int ID)const
{
	for(int index=0; index < SpeedCount(); index++)
	{
		if (m_configuration->Speeds[index].ID == ID) return index;
	}
	return -1;
}

CString CCRColorimeter::SpeedName(int Index) const
{
	return m_configuration->Speeds[Index].Name;
}

int CCRColorimeter::SyncModeCount() const
{
	return (int)m_configuration->SyncModes.GetCount();
}

int CCRColorimeter::SyncModeIDFromName(CString Name) const
{
	int Id = -1;
	for(int i=0; i < SyncModeCount(); i++)
	{
		if (m_configuration->SyncModes[i].Name == Name) 
		{
			Id = m_configuration->SyncModes[i].ID;
			break;
		}
	}
	return Id;
}

int CCRColorimeter::SyncModeIndexFromID(int ID)const
{
	for(int index=0; index < SyncModeCount(); index++)
	{
		if (m_configuration->SyncModes[index].ID == ID) return index;
	}
	return -1;
}

int CCRColorimeter::SyncModeID(int Index) const
{
	return m_configuration->SyncModes[Index].ID;
}

int CCRColorimeter::SyncMode() const
{
	return SyncModeIndexFromID(m_setupModified->SyncModeID);
}

void CCRColorimeter::SetSyncMode(int value)
{
	if (value >= 0) 
	{
        m_setupModified->SyncModeID = m_configuration->SyncModes[value].ID;
	}
}

CString CCRColorimeter::SyncModeName(int Index) const
{
	return m_configuration->SyncModes[Index].Name;
}

float CCRColorimeter::MinExposure() const
{
	return  m_configuration->MinExposure;
}

float CCRColorimeter::MaxExposure() const
{
	return m_configuration->MaxExposure;
}

float CCRColorimeter::Exposure() const
{
	return m_setupModified->Exposure;
}

void CCRColorimeter::SetExposure(float value)
{
	if (value >= MinExposure() && value <= MaxExposure()) 
	{
		m_setupModified->Exposure = value;
	}
}

float CCRColorimeter::MaxAutoExposure() const
{
	return m_setupModified->MaxAutoExposure;
}

void CCRColorimeter::SetMaxAutoExposure(float value)
{
	if (value >= MinExposure() && value <= MaxExposure()) 
	{
		m_setupModified->MaxAutoExposure = value;
	}
}
float CCRColorimeter::MinSyncFreq() const
{
	return m_configuration->MinSyncFreq;
}

float CCRColorimeter::MaxSyncFreq() const
{
	return m_configuration->MaxSyncFreq;
}

float CCRColorimeter::SyncFreq() const
{
	return m_setupModified->SyncFreq;
}

void CCRColorimeter::SetSyncFreq(float value)
{
	if (value >= MinSyncFreq() && value <= MaxSyncFreq()) 
	{
        m_setupModified->SyncFreq = value;
	}
}

int CCRColorimeter::MinExposureX() const
{
	return m_configuration->MinExposureX;
}

int CCRColorimeter::MaxExposureX() const
{
	return m_configuration->MaxExposureX;
}

int CCRColorimeter::ExposureX() const
{
	return m_setupModified->ExposureX;
}

void CCRColorimeter::SetExposureX(int value)
{
	if(value >= MinExposureX() && value <= MaxExposureX())
	{
		m_setupModified->ExposureX = value;
	}
}

// This property is obsolete. Use UserCalibModeCount instead.
int CCRColorimeter::MatrixModeCount() const
{
	return (int)m_configuration->MatrixModes.GetCount();
}

// This function is obsolete. Use UserCalibModeIDFromName instead.
int CCRColorimeter::MatrixModeIDFromName(CString Name) const
{
	int Id = -1;
	for(int i=0; i < MatrixModeCount(); i++)
	{
		if (m_configuration->MatrixModes[i].Name == Name) 
		{
			Id = m_configuration->MatrixModes[i].ID;
			break;
		}
	}
	return Id;
}

// This function is obsolete. Use UserCalibModeIndexFromID instead.
int CCRColorimeter::MatrixModeIndexFromID(int ID)const
{
	for(int index=0; index < MatrixModeCount(); index++)
	{
		if (m_configuration->MatrixModes[index].ID == ID) return index;
	}
	return -1;
}


// This function is obsolete. Use UserCalibModeID instead.
int CCRColorimeter::MatrixModeID(int Index) const
{
	return m_configuration->MatrixModes[Index].ID;
}

// This property is obsolete. Use UserCalibModeIDFromName instead.
int CCRColorimeter::MatrixMode() const
{
	return MatrixModeIndexFromID(m_setupModified->MatrixModeID);
}

// This property is obsolete. Use SetUserCalibMode instead.
void CCRColorimeter::SetMatrixMode(int value)
{
	if (value >= 0) 
	{
        m_setupModified->MatrixModeID = m_configuration->MatrixModes[value].ID;
	}
	
}
// This property is obsolete. Use UserCalibModeName instead.
CString CCRColorimeter::MatrixModeName(int Index) const
{
	return m_configuration->MatrixModes[Index].Name;
}

int CCRColorimeter::UserCalibModeCount() const
{
	return (int)m_configuration->UserCalibModes.GetCount();
}

int CCRColorimeter::UserCalibModeIDFromName(CString Name) const
{
	int Id = -1;
	for(int i=0; i < UserCalibModeCount(); i++)
	{
		if (m_configuration->UserCalibModes[i].Name == Name) 
		{
			Id = m_configuration->UserCalibModes[i].ID;
			break;
		}
	}
	return Id;
}

int CCRColorimeter::UserCalibModeIndexFromID(int ID)const
{
	for(int index=0; index < UserCalibModeCount(); index++)
	{
		if (m_configuration->UserCalibModes[index].ID == ID) return index;
	}
	return -1;
}

int CCRColorimeter::UserCalibModeID(int Index) const
{
	return m_configuration->UserCalibModes[Index].ID;
}

int CCRColorimeter::UserCalibMode() const
{
	return UserCalibModeIndexFromID(m_setupModified->UserCalibModeID);
}

void CCRColorimeter::SetUserCalibMode(int value)
{
	if (value >= 0) 
	{
        m_setupModified->UserCalibModeID = m_configuration->UserCalibModes[value].ID;
	}
}

CString CCRColorimeter::UserCalibModeName(int Index) const
{
	return m_configuration->UserCalibModes[Index].Name;
}

int CCRColorimeter::MatrixCount(int accessory) const
{
	if (accessory >= 0 && accessory < AccessoryCount()) 
	{
		return (int)m_configuration->Accessories[accessory].Matrices.GetCount();
	}
	return 0;
}

int CCRColorimeter::MatrixID(int accessory, int Index) const
{
	return m_configuration->Accessories[accessory].Matrices[Index].ID;
}

int CCRColorimeter::Matrix(int accessory) const
{
	return MatrixIndexFromID(accessory, m_setupModified->MatrixID);
}

void CCRColorimeter::SetMatrix(int accessory, int value)
{
	if (value >= 0 && value < MatrixCount(accessory)) 
	{
		m_setupModified->MatrixID = m_configuration->Accessories[accessory].Matrices[value].ID;
	}
}

int CCRColorimeter::MatrixIDFromName(int accessory, CString Name) const
{
	int Id = -1;

	for(int i = 0;i< MatrixCount(accessory); i++)
	{
		if (m_configuration->Accessories[accessory].Matrices[i].Name == Name) 
		{
			Id = m_configuration->Accessories[accessory].Matrices[i].ID;
			break;
		}
	}
	return Id;
}

int CCRColorimeter::MatrixIndexFromID(int accessory, int ID) const
{
	for(int index=0; index < MatrixCount(accessory); index++)
	{
		if(m_configuration->Accessories[accessory].Matrices[index].ID == ID) return index;
	}
	return -1;
}

CString CCRColorimeter::MatrixName(int accessory, int Index) const
{
	if (accessory >= 0 && accessory < AccessoryCount() && Index >= 0 && Index < MatrixCount(accessory)) 
	{
		return m_configuration->Accessories[accessory].Matrices[Index].Name;
	}
	else
	{
		return _T("None");
	}
}

int CCRColorimeter::MatchCount() const
{
	return (int)m_configuration->MatchSet.GetCount();
}

int CCRColorimeter::MatchID(int Index) const
{
	return m_configuration->MatchSet[Index].ID;
}

int CCRColorimeter::Match() const
{
	return MatchIndexFromID(m_setupModified->MatchID);
}

void CCRColorimeter::SetMatch(int value)
{
	if (value >= 0 && value < MatchCount()) 
	{
		m_setupModified->MatchID = m_configuration->MatchSet[value].ID;
	}
}

int CCRColorimeter::MatchIDFromName(CString Name) const
{
	int Id = -1;

	for(int i = 0;i< MatchCount(); i++)
	{
		if (m_configuration->MatchSet[i].Name == Name) 
		{
			Id = m_configuration->MatchSet[i].ID;
			break;
		}
	}
	return Id;
}

int CCRColorimeter::MatchIndexFromID(int ID) const
{
	for(int index=0; index < MatchCount(); index++)
	{
		if(m_configuration->MatchSet[index].ID == ID) return index;
	}
	return -1;
}

CString CCRColorimeter::MatchName(int Index) const
{
	if (Index >= 0 && Index < MatchCount()) 
	{
		return m_configuration->MatchSet[Index].Name;
	}
	else
	{
		return _T("None");
	}
}

float CCRColorimeter::MinSamplingRate() const
{
	return m_configuration->MinSamplingRate;
}

float CCRColorimeter::MaxSamplingRate() const
{
	return m_configuration->MaxSamplingRate;
}

float CCRColorimeter::SamplingRate() const
{
	return m_setupModified->SamplingRate;
}

void CCRColorimeter::SetSamplingRate(float value)
{
	if (value >= MinSamplingRate() && value <= MaxSamplingRate()) 
	{
        m_setupModified->SamplingRate = value;
	}
}


int CCRColorimeter::FlickerFilterType() const
{
	return m_flickerFilter.filter_type;
}

void CCRColorimeter::SetFlickerFilterType(int value)
{
	 m_flickerFilter.filter_type = (uint8_t)value;
}



CString CCRColorimeter::FlickerFilterTypeName() const
{
	switch(m_flickerFilter.filter_type)
	{
	case 1:
		return _T("Low Pass");
	case 2:
		return _T("High Pass");
	case 3:
		return _T("Band Pass");
	case 4:
		return _T("Band Stop");
	case 0:
	default:
		return _T("None");
	}
}


int CCRColorimeter::FlickerFilterFamily() const
{
	return m_flickerFilter.filter_family;
}

void CCRColorimeter::SetFlickerFilterFamily(int value)
{
	 m_flickerFilter.filter_family = (uint8_t)value;
}


int CCRColorimeter::FlickerFilterOrder() const
{
	return m_flickerFilter.order;
}

void CCRColorimeter::SetFlickerFilterOrder(int value)
{
	 m_flickerFilter.order = (uint8_t)value;
}


double CCRColorimeter::FlickerFilterFrequency() const
{
	return m_flickerFilter.frequency;
}

void CCRColorimeter::SetFlickerFilterFrequency(double value)
{
	 m_flickerFilter.bandwidth = value;
}
   

double CCRColorimeter::FlickerFilterBandwidth() const
{
	return m_flickerFilter.bandwidth;
}

void CCRColorimeter::SetFlickerFilterBandwidth(double value)
{
	 m_flickerFilter.frequency = value;
}


double CCRColorimeter::FlickerMaxSearchFrequency() const
{
	return m_flickerMaxSearchFrequency;
}

void CCRColorimeter::SetFlickerMaxSearchFrequency(double value)
{
	 m_flickerMaxSearchFrequency = value;
}


int CCRColorimeter::ResponseTimeFilterType() const
{
	return m_responseTimeSettings.filter_type;
}

void CCRColorimeter::SetResponseTimeFilterType(int value)
{
	 m_responseTimeSettings.filter_type = (uint8_t)value;
}



CString CCRColorimeter::ResponseTimeFilterTypeName() const
{
	switch(m_responseTimeSettings.filter_type)
	{
	case 1:
		return _T("Moving Window Average");
	case 0:
	default:
		return _T("None");
	}
}

int CCRColorimeter::ResponseTimeMode() const
{
	return m_responseTimeSettings.mode;
}

void CCRColorimeter::SetResponseTimeMode(int value)
{	
	 m_responseTimeSettings.mode = (uint8_t)value;
}
	
int CCRColorimeter::ResponseTimeAverage() const
{
	return m_responseTimeSettings.average;
}

void CCRColorimeter::SetResponseTimeAverage(int value)
{	
	 m_responseTimeSettings.average = (uint8_t)value;
}

		
BOOL CCRColorimeter::ResponseTimeClippingEnabled() const
{
	return m_responseTimeSettings.clipping_enabled;
}

void CCRColorimeter::SetResponseTimeClippingEnabled(BOOL enabled)
{
	
	 m_responseTimeSettings.clipping_enabled = (uint8_t)enabled;
}

		
float CCRColorimeter::ResponseTimeClippingLowerLimit() const
{	
	return m_responseTimeSettings.clipping_lo;
}

void CCRColorimeter::SetResponseTimeClippingLowerLimit(float value)
{
	 m_responseTimeSettings.clipping_lo = (double)value;
}
		
float CCRColorimeter::ResponseTimeClippingUpperLimit() const
{	
	return m_responseTimeSettings.clipping_hi;
}

void CCRColorimeter::SetResponseTimeClippingUpperLimit(float value)
{
	 m_responseTimeSettings.clipping_hi = (double)value;
}
	
float CCRColorimeter::ResponseTimeNoiseLevel() const
{	
	return m_responseTimeSettings.noiselevel;
}

void CCRColorimeter::SetResponseTimeNoiseLevel(float value)
{
	 m_responseTimeSettings.noiselevel = (double)value;
}

float CCRColorimeter::ResponseTimeStepResponseZoneLowerLimit() const
{
	return m_responseTimeSettings.setupresponsezone_lo;
}

void CCRColorimeter::SetResponseTimeStepResponseZoneLowerLimit(float value)
{
	 m_responseTimeSettings.setupresponsezone_lo = (double)value;
}

float CCRColorimeter::ResponseTimeStepResponseZoneUpperLimit() const
{
	return m_responseTimeSettings.setupresponsezone_hi;
}

void CCRColorimeter::SetResponseTimeStepResponseZoneUpperLimit(float value)
{
	 m_responseTimeSettings.setupresponsezone_hi = (double)value;

}

int CCRColorimeter::MaxCMF() const
{
	return 4;
}

int CCRColorimeter::CMF() const
{
	return m_setupModified->CMF;
}

void CCRColorimeter::SetCMF(int value)
{
	if (value >= 0 && value < MaxCMF()) 
	{
		m_setupModified->CMF = value;
	}
}

CString CCRColorimeter::Reading(CString DataType) const
{
	if(DataType == _T("ID"))
		return m_reading->ID;
	else if(DataType == _T("Model"))
		return m_reading->Model;
	else if(DataType == _T("Time"))
	return m_reading->Time;
	else if(DataType == _T("Accessory"))
		return m_reading->Accessory;
	else if(DataType == _T("Filter"))
		return m_reading->Filter;
	else if(DataType == _T("Aperture"))
		return m_reading->Aperture;
	else if(DataType == _T("Mode"))
		return m_reading->Mode;
	else if(DataType == _T("ExposureMode"))
		return m_reading->ExposureMode;
	else if(DataType == _T("Exposure"))
		return m_reading->Exposure;
	else if(DataType == _T("MaxAutoExposure"))
		return m_reading->MaxAutoExposure;
	else if(DataType == _T("RangeMode"))
		return m_reading->RangeMode;
	else if(DataType == _T("Range"))
		return m_reading->Range;
	else if(DataType == _T("Speed"))
		return m_reading->Speed;
	else if(DataType == _T("SyncMode"))
		return m_reading->SyncMode;
	else if(DataType == _T("SyncFreq"))
		return m_reading->SyncFreq;
	else if(DataType == _T("ExposureX"))
		return m_reading->ExposureX;
	else if(DataType == _T("MatrixMode"))
		return m_reading->MatrixMode;
	else if(DataType == _T("UserCalibMode"))
		return m_reading->UserCalibMode;
	else if(DataType == _T("Matrix"))
	{
		int accessory  = -1;
		int matrix = -1;
		accessory = AccessoryIndexFromID(AccessoryIDFromName(m_reading->Accessory));
		matrix = MatrixIndexFromID(accessory, _ttoi(m_reading->MatrixID));
		return MatrixName(accessory, matrix);
	}
	else if(DataType == _T("Match"))
	{
		int match = -1;
		match = MatchIndexFromID(_ttoi(m_reading->MatchID));
		return MatchName(match);
	}
	else if(DataType == _T("X"))
		return m_reading->CIE[Observer_2Degree].X;
	else if(DataType == _T("Y"))
		return m_reading->CIE[Observer_2Degree].Y;
	else if(DataType == _T("Z"))
		return m_reading->CIE[Observer_2Degree].Z;
	else if(DataType == _T("XYZ"))
		return m_reading->CIE[Observer_2Degree].XYZ;
	else if(DataType == _T("xy"))
		return m_reading->CIE[Observer_2Degree].xy;
	else if(DataType == _T("uv"))
		return m_reading->CIE[Observer_2Degree].uv;
	else if(DataType == _T("upvp"))
		return m_reading->CIE[Observer_2Degree].upvp;
	else if(DataType == _T("CCT"))
		return m_reading->CIE[Observer_2Degree].CCT;
	else if(DataType == _T("Yv"))
		return m_reading->Yv;
	else if(DataType == _T("Radiometric"))
		return m_reading->Radiometric;
	else if(DataType == _T("CMF"))
		return m_reading->CMF;
	else if(DataType == _T("Warnings"))
		return m_reading->AllWarnings;
	else if(DataType == _T("StartWavelength"))
	{
		CString floatString;
		floatString.Format(_T("%f"),m_reading->Spectrum.StartingWavelength);
		return floatString;
	}
	else if(DataType == _T("EndWavelength"))
	{
		CString floatString;
		floatString.Format(_T("%f"),m_reading->Spectrum.EndingWavelength);
		return floatString;
	}
	else if(DataType == _T("DeltaWavelength"))
	{
		CString floatString;
		floatString.Format(_T("%f"),m_reading->Spectrum.Delta);
		return floatString;
	}
	else if(DataType == _T("X10"))
		return m_reading->CIE[Observer_10Degree].X;
	else if(DataType == _T("Y10"))
		return m_reading->CIE[Observer_10Degree].Y;
	else if(DataType == _T("Z10"))
		return m_reading->CIE[Observer_10Degree].Z;
	else if(DataType == _T("XYZ10"))
		return m_reading->CIE[Observer_10Degree].XYZ;
	else if(DataType == _T("xy10"))
		return m_reading->CIE[Observer_10Degree].xy;
	else if(DataType == _T("SamplingRate"))
	{
		CString floatString;
		floatString.Format(_T("%f"),m_reading->Temporal.SamplingRate);
		return floatString;
	}
	return CString();
}

CArray<double,double>& CCRColorimeter::Spectrum() const
{
	return m_reading->Spectrum.Data;
}

CArray<double,double>& CCRColorimeter::Temporal() const
{
	return m_reading->Temporal.Data;
}

BOOL CCRColorimeter::UploadSetup()
{
	if(m_setupModified->AccessoryID != m_setup->AccessoryID){
		SendCommand(CStringFormat(_T("SM Accessory %d"), m_setupModified->AccessoryID));
			//SendCommand(CStringFormat(_T("RC Matrix %d"), m_setupModified->AccessoryID)) 'Download the matrix for the current accessory
	}
	if(m_setupModified->Filter1ID != m_setup->Filter1ID){
		SendCommand(CStringFormat(_T("SM Filter1 %d"), m_setupModified->Filter1ID));
	}
	if(m_setupModified->Filter2ID != m_setup->Filter2ID){
		SendCommand(CStringFormat(_T("SM Filter2 %d"), m_setupModified->Filter2ID));
	}
	if(m_setupModified->Filter3ID != m_setup->Filter3ID){
		SendCommand(CStringFormat(_T("SM Filter3 %d"), m_setupModified->Filter3ID));
	}
	if(m_setupModified->ApertureID != m_setup->ApertureID){
		SendCommand(CStringFormat(_T("SM Aperture %d"), m_setupModified->ApertureID));
	}
	if(m_setupModified->ModeID != m_setup->ModeID){
		SendCommand(CStringFormat(_T("SM Mode %d"), m_setupModified->ModeID));
	}
	if(m_setupModified->ExposureModeID != m_setup->ExposureModeID){
		SendCommand(CStringFormat(_T("SM ExposureMode %d"), m_setupModified->ExposureModeID));
	}
	if(m_setupModified->Exposure != m_setup->Exposure){
		SendCommand(CStringFormat(_T("SM Exposure %f"), m_setupModified->Exposure));
	}
	if(m_setupModified->MaxAutoExposure != m_setup->MaxAutoExposure){
		SendCommand(CStringFormat(_T("SM MAxAutoExposure %f"), m_setupModified->MaxAutoExposure));
	}
	if(m_setupModified->RangeModeID != m_setup->RangeModeID){
		SendCommand(CStringFormat(_T("SM RangeMode %d"), m_setupModified->RangeModeID));
	}
	if(m_setupModified->RangeID != m_setup->RangeID){
		SendCommand(CStringFormat(_T("SM Range %d"), m_setupModified->RangeID));
	}
	if(m_setupModified->SpeedID != m_setup->SpeedID){
		SendCommand(CStringFormat(_T("SM Speed %d"), m_setupModified->SpeedID));
	}
	if(m_setupModified->SyncModeID != m_setup->SyncModeID){
		SendCommand(CStringFormat(_T("SM SyncMode %d"), m_setupModified->SyncModeID));
	}
	if(m_setupModified->SyncFreq != m_setup->SyncFreq){
		SendCommand(CStringFormat(_T("SM SyncFreq %f"), m_setupModified->SyncFreq));
	}
	if(m_setupModified->ExposureX != m_setup->ExposureX){
		SendCommand(CStringFormat(_T("SM ExposureX %d"), m_setupModified->ExposureX));
	}
	//if(m_setupModified->MatrixModeID != m_setup->MatrixModeID){
	//   SendCommand(CStringFormat(_T("SM MatrixMode %d"), m_setupModified->MatrixModeID));
	//}
	if(m_setupModified->UserCalibModeID != m_setup->UserCalibModeID){
		SendCommand(CStringFormat(_T("SM UserCalibMode %d"), m_setupModified->UserCalibModeID));
	}
	if(m_setupModified->MatrixID != m_setup->MatrixID){
		SendCommand(CStringFormat(_T("SM Matrix %d"), m_setupModified->MatrixID));
	}
	if(m_setupModified->MatchID != m_setup->MatchID){
		SendCommand(CStringFormat(_T("SM Match %d"), m_setupModified->MatchID));
	}
	if(m_setupModified->SamplingRate != m_setup->SamplingRate){
		SendCommand(CStringFormat(_T("SM SamplingRate %f"), m_setupModified->SamplingRate));
	}
	if(m_setupModified->CMF != m_setup->CMF){
		SendCommand(CStringFormat(_T("SM CMF %d"), m_setupModified->CMF));
	}
	return TRUE;
}

 BOOL CCRColorimeter::SendCommand(CString command, int Timeout)
 {
	 TRACE(_T("Start Sending Message\n"));
	 //SyncLock _messageLock

	 m_channel->SetReadTimeout(Timeout);
	 command.Append(NEW_LINE);
	 m_channel->WriteLine(command);
	 CString message;
	 TRACE(_T("Data Transmitted: %s\n"), command);
	 Sleep(20);
	 m_commandQ.Push(command);

	//::SendMessage(m_pOwner->m_hWnd, UWM_DATA_DEBUG, (WPARAM)(LPCTSTR)command, (LPARAM) this);

	 TRACE(_T("Finished Sending Message\n"));
	 //End SyncLock

	return TRUE;
 }

 void CCRColorimeter::DataReceivedNotifier::OnNotifcation(NotificationType type, const CString& buffer)
 {
	 if(type == DataRead)
		 m_meter->DataReceivedHandler(buffer);
	 else if(type == DataWritten)
		 m_meter->DataTransmittedHandler(buffer);
 }

void CCRColorimeter::DataReceivedHandler(const CString& buffer)
{
	m_mutex.Lock();
	CString indata =  m_channel->ReadExisting();

	TRACE(_T("Data Received:%s\n"), indata);

	m_buffer.Append(indata);

	TRACE(_T("DataReceived signaled.\n"));

	TRACE(_T("Finish Receiving Response\n"));

	m_mutex.Unlock();

	ProcessBuffer();
	ProcessResponses();
}

void CCRColorimeter::DataTransmittedHandler(const CString& buffer)
{
	::SendMessage(m_pOwner->m_hWnd, UWM_DATA_SENT, (WPARAM)(LPCTSTR)buffer, (LPARAM) this);
}


void CCRColorimeter::ProcessBuffer()
{
	TRACE(_T("Start Processing Buffer\n"));
	BOOL foundSomethingToProcess = FALSE;

	int position = -1;
	m_mutex.Lock();
	position = m_buffer.Find(NEW_LINE);
	m_mutex.Unlock();
	while (position > 0)
	{
		//CString messageText = m_buffer.SpanExcluding(NEW_LINE);
		//m_buffer = m_buffer.Right(m_buffer.GetLength() - (messageText.GetLength() + CString(NEW_LINE).GetLength()));

		CString messageText;
		m_mutex.Lock();
		messageText = m_buffer.Left(position + CString(NEW_LINE).GetLength());
		m_buffer = m_buffer.Right(m_buffer.GetLength() - messageText.GetLength());
		m_mutex.Unlock();
		if (messageText.GetAt(0) == '>') {			
			CString errorMessage = CStringFormat(_T("DISCARDED:%s"), messageText);
			AfxMessageBox(errorMessage); // This has an echo prompt so discard this message
			::SendMessage(m_pOwner->m_hWnd, UWM_DATA_ERROR, (WPARAM)(LPCTSTR)errorMessage, (LPARAM) this);
		}
		else if (m_commandQ.GetCount() > 0 && m_commandQ.GetHead() == messageText) {	
			CString errorMessage = CStringFormat(_T("DISCARDED:%s"), messageText);
			AfxMessageBox(errorMessage); // The first response does not have an echo prompt, but discard if it's the same as the message sent
			::SendMessage(m_pOwner->m_hWnd, UWM_DATA_ERROR, (WPARAM)(LPCTSTR)errorMessage, (LPARAM) this);	
		}
		else {

			m_responseQ.Push(messageText);
			if (messageText.Left(2) == _T("ER")) {
				::SendMessage(m_pOwner->m_hWnd, UWM_DATA_ERROR, (WPARAM)(LPCTSTR)messageText, (LPARAM) this); // Error report
			}
			else {
				::SendMessage(m_pOwner->m_hWnd, UWM_DATA_RECEIVED, (WPARAM)(LPCTSTR)messageText, (LPARAM) this);
			}
		}
		position = -1;
		m_mutex.Lock();
		position = m_buffer.Find(NEW_LINE);
		m_mutex.Unlock();

	}

	TRACE(_T("Finished Processing Buffer\n"));
 }

void CCRColorimeter::FlushBuffer()
{
	if (m_buffer.GetLength() != 0) {
		//::SendMessage DataIgnored(m_buffer);
	}
}

void CCRColorimeter::ProcessResponses()
{
	if (m_responseQ.GetCount() == 0)
		return;

	TRACE(_T("Start Processing Response\n"));
	BOOL canProcess = TRUE; //Continue processing responses
	int lines = 0; //Number of lines the response contains
	int itemLines = 0; //Number of lines to read to finish processing the command
	int processed = FALSE; //Finished Processing the command, Pop Command and Lines of Response

	CString type = _T(""); //For Eg. OK Or ER"
	CString typeCode = _T(""); //For Eg. -510
	CString command = _T(""); // For Eg. RS Accessory
	CString result = _T(""); //Could be number of lines, Result or Error/Warning Description
	do
	{
		processed = FALSE;
		lines = 0;

		CString message = m_commandQ.GetHead();
		message.Remove('\n');
		message.Remove('\r');
		CString response = m_responseQ.GetHead();
		response.Remove('\n');
		response.Remove('\r');
		CStringArray words;
		Split(words, response, RESPONSE_SEPARATOR);

		for(int wordIndex = 0; wordIndex < words.GetCount(); wordIndex++)
		{
			switch(wordIndex)
			{
			case 0:
				type = words[wordIndex];
				break;
			case 1:
				typeCode = words[wordIndex];
				break;
			case 2:
				command = words[wordIndex];
				break;
			case 3:
				result = words[wordIndex];
				break;
			default:
				//Invalid command: Has extra words
				break;
			}
		}

		if (! command.IsEmpty()) 
		{
			// This is a valid response now match the response to the message
			if(message.Left(command.GetLength()) == command)
			{
				lines += 1;
			}
			else
			{
				//response does not match with the message				
				AfxMessageBox(CStringFormat(_T("Command [%s] does not match with the response [%s]"), message, response));				
				AfxThrowUserException();
			}
		}

		if (type == RESPONSE_ERROR) {
			processed = TRUE;
		}
		else if (type == RESPONSE_OK) {
			// Successful message

			//------------Read Configuration------------
			if(command == _T("RC Firmware")){
				m_configuration->Firmware = result;
				processed = TRUE;
			}
			else if(command == _T("RC ID")){
				m_configuration->ID = result;
				processed = TRUE;
			}
			else if(command == _T("RC Model")){
				m_configuration->Model = result;
				processed = TRUE;
			}
			else if(command == _T("RC Accessory")){

				if(!_stscanf_s(result, _T("%d"), &itemLines)) {					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}

				lines += itemLines;
				m_configuration->AccessoryCount = itemLines;
				m_configuration->Accessories.SetSize(itemLines);
				if (m_responseQ.GetCount() > itemLines) {
					for(int index=1; index<=itemLines; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 3) { 
							
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));
							
							AfxThrowUserException();
						}

						if (!_stscanf_s(items[0], _T("%d"), &m_configuration->Accessories[index - 1].ID)) {
							AfxMessageBox(_T("Id is not a valid number"));
							AfxThrowUserException();
						}

						m_configuration->Accessories[index - 1].Matrices.SetSize(0);// initialize the matrices to zero
						m_configuration->Accessories[index - 1].Name = items[1];
						m_configuration->Accessories[index - 1].Type = items[2];

						// After the accessory id's are retrieved send the matrix and matrix calibration commands here
						CString strCommand;
						strCommand.Format(_T("RC Matrix %d"), m_configuration->Accessories[index - 1].ID);
						SendCommand(strCommand);

						//SendCommand(String.Format("RC MatrixCalibration {0}"), m_configuration->Accessories[index - 1].ID))
					}
					processed = TRUE;

				}
			}
			else if(command == _T("RC Filter")){
				if(!_stscanf_s(result, _T("%d"), &itemLines)) {
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					AfxThrowUserException();
				}
				lines += itemLines;
				m_configuration->FilterCount = itemLines;
				m_configuration->Filters.SetSize(itemLines);
				if (m_responseQ.GetCount() > itemLines) {
					for(int index=1; index<=itemLines; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 3) { 
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));
							AfxThrowUserException();
						}

						if (!_stscanf_s(items[0], _T("%d"), &m_configuration->Filters[index - 1].ID)) {
							AfxMessageBox(_T("Id is not a valid number"));
							AfxThrowUserException();

						}
						m_configuration->Filters[index - 1].Name = items[1];
						m_configuration->Filters[index - 1].Type = items[2];

					}
					processed = TRUE;
				}
			}
			else if(command == _T("RC Aperture")){
				if(!_stscanf_s(result, _T("%d"), &itemLines)) {
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					AfxThrowUserException();
				}
				lines += itemLines;
				m_configuration->ApertureCount = itemLines;
				m_configuration->Apertures.SetSize(itemLines);
				if (m_responseQ.GetCount() > itemLines) {
					for(int index=1; index<=itemLines; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 2) { 
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));
							AfxThrowUserException();
						}

						if (!_stscanf_s(items[0], _T("%d"), &m_configuration->Apertures[index - 1].ID)) {
							AfxMessageBox(_T("Id is not a valid number"));
							AfxThrowUserException();
						}
						m_configuration->Apertures[index - 1].Name = items[1];

					}
					processed = TRUE;
				}
			}
			else if(command == _T("RC Mode")) {
				if(!_stscanf_s(result, _T("%d"), &itemLines)) {
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					AfxThrowUserException();
				}
				lines += itemLines;
				m_configuration->ModesCount = itemLines;
				m_configuration->Modes.SetSize(itemLines);
				if (m_responseQ.GetCount() > itemLines) {
					for(int index=1; index<=itemLines; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 2) { 							
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));
							AfxThrowUserException();
						}

						if (!_stscanf_s(items[0], _T("%d"), &m_configuration->Modes[index - 1].ID)) {
							AfxMessageBox(_T("Id is not a valid number"));
							AfxThrowUserException();

						}
						m_configuration->Modes[index - 1].Name = items[1];

					}
					processed = TRUE;
				}
			}
			else if(command == _T("RC ExposureMode")) {
				if(!_stscanf_s(result, _T("%d"), &itemLines)) {					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));					
					AfxThrowUserException();
				}
				lines += itemLines;
				m_configuration->ExposureModesCount = itemLines;
				m_configuration->ExposureModes.SetSize(itemLines);
				if (m_responseQ.GetCount() > itemLines) {
					for(int index=1; index<=itemLines; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 2) { 							
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));							
							AfxThrowUserException();
						}

						if (!_stscanf_s(items[0], _T("%d"), &m_configuration->ExposureModes[index - 1].ID)) {
							AfxMessageBox(_T("Id is not a valid number"));
							AfxThrowUserException();

						}
						m_configuration->ExposureModes[index - 1].Name = items[1];

					}
					processed = TRUE;
				}
			}
			else if(command == _T("RC RangeMode")) {
				if(!_stscanf_s(result, _T("%d"), &itemLines)) {					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));					
					AfxThrowUserException();
				}
				lines += itemLines;
				
				m_configuration->RangeModesCount = itemLines;
				m_configuration->RangeModes.SetSize(itemLines);
				if (m_responseQ.GetCount() > itemLines) {
					for(int index=1; index<=itemLines; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 2) { 							
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));							
							AfxThrowUserException();
						}

						if (!_stscanf_s(items[0], _T("%d"), &m_configuration->RangeModes[index - 1].ID)) {
							AfxMessageBox(_T("Id is not a valid number"));
							AfxThrowUserException();

						}
						m_configuration->RangeModes[index - 1].Name = items[1];

					}
					processed = TRUE;
				}
			}
			else if(command == _T("RC Range")) {
				if(!_stscanf_s(result, _T("%d"), &itemLines)) {					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));					
					AfxThrowUserException();
				}
				lines += itemLines;
				m_configuration->RangesCount = itemLines;
				m_configuration->Ranges.SetSize(itemLines);
				if (m_responseQ.GetCount() > itemLines) {
					for(int index=1; index<=itemLines; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 2) { 
							
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));
							
							AfxThrowUserException();
						}

						if (!_stscanf_s(items[0], _T("%d"), &m_configuration->Ranges[index - 1].ID)) {
							AfxMessageBox(_T("Id is not a valid number"));
							AfxThrowUserException();

						}
						m_configuration->Ranges[index - 1].Name = items[1];

					}
					processed = TRUE;
				}
			}
			else if(command == _T("RC Speed")) {
				if(!_stscanf_s(result, _T("%d"), &itemLines)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				lines += itemLines;
				m_configuration->SpeedsCount = itemLines;
				m_configuration->Speeds.SetSize(itemLines);
				if (m_responseQ.GetCount() > itemLines) {
					for(int index=1; index<=itemLines; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 2) { 
							
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));
							
							AfxThrowUserException();
						}

						if (!_stscanf_s(items[0], _T("%d"), &m_configuration->Speeds[index - 1].ID)) {
							AfxMessageBox(_T("Id is not a valid number"));
							AfxThrowUserException();

						}
						m_configuration->Speeds[index - 1].Name = items[1];

					}
					processed = TRUE;
				}
			}
			else if(command == _T("RC SyncMode")) {
				if(!_stscanf_s(result, _T("%d"), &itemLines)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				lines += itemLines;
				m_configuration->SyncModesCount = itemLines;
				m_configuration->SyncModes.SetSize(itemLines);
				if (m_responseQ.GetCount() > itemLines) {
					for(int index=1; index<=itemLines; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 2) { 
							
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));
							
							AfxThrowUserException();
						}

						if (!_stscanf_s(items[0], _T("%d"), &m_configuration->SyncModes[index - 1].ID)) {
							AfxMessageBox(_T("Id is not a valid number"));
							AfxThrowUserException();

						}
						m_configuration->SyncModes[index - 1].Name = items[1];

					}
					processed = TRUE;
				}
			}
			else if(command == _T("RC MatrixMode")) {
				if(!_stscanf_s(result, _T("%d"), &itemLines)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				lines += itemLines;
				//m_configuration->MatarixModesCount = itemLines;
				m_configuration->MatrixModes.SetSize(itemLines);
				if (m_responseQ.GetCount() > itemLines) {
					for(int index=1; index<=itemLines; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 2) { 
							
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));
							
							AfxThrowUserException();
						}

						if (!_stscanf_s(items[0], _T("%d"), &m_configuration->MatrixModes[index - 1].ID)) {
							AfxMessageBox(_T("Id is not a valid number"));
							AfxThrowUserException();

						}
						m_configuration->MatrixModes[index - 1].Name = items[1];

					}
					processed = TRUE;
				}
			}
			else if(command == _T("RC UserCalibMode")) {
				if(!_stscanf_s(result, _T("%d"), &itemLines)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				lines += itemLines;
				//m_configuration->UserCalibModesCount = itemLines;
				m_configuration->UserCalibModes.SetSize(itemLines);
				if (m_responseQ.GetCount() > itemLines) {
					for(int index=1; index<=itemLines; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 2) { 
							
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));
							
							AfxThrowUserException();
						}

						if (!_stscanf_s(items[0], _T("%d"), &m_configuration->UserCalibModes[index - 1].ID)) {
							AfxMessageBox(_T("Id is not a valid number"));
							AfxThrowUserException();

						}
						m_configuration->UserCalibModes[index - 1].Name = items[1];

					}
					processed = TRUE;
				}
			}
			else if(command == _T("RC Matrix")) {
				CStringArray resultItems;
				Split(resultItems, result, RESULT_SEPARATOR);
				int accessory = 0;
				if (resultItems.GetCount() != 2) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				if (resultItems[0] == "None") {
					itemLines = 0;
					processed = TRUE;
				}
				else{
					if(!_stscanf_s(resultItems[0], _T("%d"), &itemLines)) {
						
						AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
						
						AfxThrowUserException();
					}
				}

				if(!_stscanf_s(resultItems[1], _T("%d"), &accessory)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				lines += itemLines;
				if (m_configuration->Accessories[accessory].Matrices.GetCount() != itemLines) { 
					m_configuration->Accessories[accessory].Matrices.SetSize(itemLines);
				}
				if (m_responseQ.GetCount() > itemLines) {
					for(int index=1; index<=itemLines; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 2) { 
							
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));
							
							AfxThrowUserException();
						}

						if (!_stscanf_s(items[0], _T("%d"), &m_configuration->Accessories[accessory].Matrices[index - 1].ID)) {
							AfxMessageBox(_T("Id is not a valid number"));
							AfxThrowUserException();

						}
						m_configuration->Accessories[accessory].Matrices[index - 1].Name = items[1];

					}
					processed = TRUE;
				}
			}
			else if(command == _T("RC MatrixCalibration")){
				CStringArray resultItems;
				Split(resultItems, result, RESULT_SEPARATOR);
				int accessory =0;
				if (resultItems.GetCount() != 2) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				if (resultItems[0] == "None") {
					itemLines = 0;
					processed = TRUE;
				}
				else{
					if(!_stscanf_s(resultItems[0], _T("%d"), &itemLines)) { 
						
						AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
						
						AfxThrowUserException();
					}
				}
				if(!_stscanf_s(resultItems[1], _T("%d"), &accessory)) { 
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				lines += itemLines;
				if (m_configuration->Accessories[accessory].Matrices.GetCount() != itemLines) { 
					m_configuration->Accessories[accessory].Matrices.SetSize(itemLines);
				}
				if (m_responseQ.GetCount() > itemLines) {
					for(int index=1; index<=itemLines; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 11) { 
							
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));
							
							AfxThrowUserException();
						}

						if (m_configuration->Accessories[accessory].Matrices[index - 1].Name != items[1]) {
							
							AfxMessageBox(CStringFormat(_T("Matrix Calibration Name does not match Matrix Name:[%s]"), itemResponse));
							
							AfxThrowUserException();
						}

						m_configuration->Accessories[accessory].Matrices[index - 1].Calibration.SetSize(9);
						for(int factor = 0; factor<8; factor++) {
							if(!_stscanf_s(items[factor + 2], _T("%f"), &m_configuration->Accessories[accessory].Matrices[index - 1].Calibration[factor])) { 
								AfxMessageBox(_T("Factor is not a valid number"));
								AfxThrowUserException();
							}
						} // factor
					} // index
					processed = TRUE;
				}
			}
			else if(command == _T("RC Match")){
				CStringArray resultItems;
				Split(resultItems, result, RESULT_SEPARATOR);

				if (resultItems.GetCount() != 1) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				if (resultItems[0] == "None") {
					itemLines = 0;
					processed = TRUE;
				}
				else{
					if(!_stscanf_s(resultItems[0], _T("%d"), &itemLines)) { 
						
						AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
						
						AfxThrowUserException();
					}
				}

				lines += itemLines;
				m_configuration->MatchCount = itemLines;
				if (m_configuration->MatchSet.GetCount() != itemLines) {
					m_configuration->MatchSet.SetSize(itemLines);
				}
				if (m_responseQ.GetCount() > itemLines) {
					for(int index=1; index<=itemLines; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 2) { 
							
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));
							
							AfxThrowUserException();
						}

						if (!_stscanf_s(items[0], _T("%d"), &m_configuration->MatchSet[index - 1].ID)) {
							AfxMessageBox(_T("Id is not a valid number"));
							AfxThrowUserException();

						}
						m_configuration->MatchSet[index - 1].Name = items[1];

					}
					processed = TRUE;
				}
			}
			else if(command == _T("RC MatchCalibration")){
				CStringArray resultItems;
				Split(resultItems, result, RESULT_SEPARATOR);
				int accessory = 0;

				if (resultItems.GetCount() != 2) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				if (resultItems[0] == "None") {
					itemLines = 0;
					processed = TRUE;
				}
				else {
					if(!_stscanf_s(resultItems[0], _T("%d"), &itemLines)) { 
						
						AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
						
						AfxThrowUserException();
					}
				}
				if(!_stscanf_s(resultItems[1], _T("%d"), &accessory)) { 
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				lines += itemLines;
				if (m_configuration->MatchSet.GetCount() != itemLines) { 
					m_configuration->MatchSet.SetSize(itemLines);
				}
				if (m_responseQ.GetCount() > itemLines) {
					for(int index=1; index<=itemLines; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 11) { 
							
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));
							
							AfxThrowUserException();
						}

						if (m_configuration->MatchSet[index - 1].Name != items[1]) {
							
							AfxMessageBox(CStringFormat(_T("Calibration Match Name does not match Match Name:[%s]"), itemResponse));
							
							AfxThrowUserException();
						}

						m_configuration->MatchSet[index - 1].Calibration.SetSize(3);
						for(int factor = 0; factor < 2; factor++) {
							if(!_stscanf_s(items[factor + 2], _T("%f"), &m_configuration->MatchSet[index - 1].Calibration[factor])) { 
								AfxMessageBox(_T("Factor is not a valid number"));
								AfxThrowUserException();
							}
						} //factor
					} //index
					processed = TRUE;
				}
			}
			else if(command == _T("RC MinExposure")){
				CStringArray items;
				Split(items, result, RESULT_SPACE);
				if (items.GetCount() != 2) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				if (!_stscanf_s(items[0], _T("%f"), &m_configuration->MinExposure)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, items[0]));
					
					AfxThrowUserException();
				}
				processed = TRUE;
			}
			else if(command == _T("RC MaxExposure")){
				CStringArray items;
				Split(items, result, RESULT_SPACE);
				if (items.GetCount() != 2) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				if (!_stscanf_s(items[0], _T("%f"), &m_configuration->MaxExposure)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, items[0]));
					
					AfxThrowUserException();

				}
				processed = TRUE;
			}
			else if(command == _T("RC MinSyncFreq")){
				CStringArray items;
				Split(items, result, RESULT_SPACE);
				if (items.GetCount() != 2) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				if (!_stscanf_s(items[0], _T("%f"), &m_configuration->MinSyncFreq)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, items[0]));
					
					AfxThrowUserException();

				}
				processed = TRUE;
			}
			else if(command == _T("RC MaxSyncFreq")){
				CStringArray items;
				Split(items, result, RESULT_SPACE);
				if (items.GetCount() != 2) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				if (!_stscanf_s(items[0], _T("%f"), &m_configuration->MaxSyncFreq)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, items[0]));
					
					AfxThrowUserException();

				}
				processed = TRUE;
			}
			else if(command == _T("RC MinExposureX")){
				if(!_stscanf_s(result, _T("%d"), &m_configuration->MinExposureX)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				processed = TRUE;
			}
			else if(command == _T("RC MaxExposureX")){
				if(!_stscanf_s(result, _T("%d"), &m_configuration->MaxExposureX)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				processed = TRUE;
			}
			else if(command == _T("RC InstrumentType")){
				if(!_stscanf_s(result, _T("%d"), &m_configuration->InstrumentType)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				processed = TRUE;
			}
			else if(command == _T("RC MinSamplingRate")){				
				if (!_stscanf_s(result, _T("%f"), &m_configuration->MinSamplingRate)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				processed = TRUE;
			}
			else if(command == _T("RC MaxSamplingRate")){
				if (!_stscanf_s(result, _T("%f"), &m_configuration->MaxSamplingRate)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();

				}
				processed = TRUE;
			}
			//------------Read Setup------------
			else if(command == _T("RS Accessory")){
				m_setup->AccessoryID = AccessoryIDFromName(result);
				m_setupModified->AccessoryID = m_setup->AccessoryID;
				processed = TRUE;
			}
			else if(command == _T("RS Filter")){
				CStringArray items;
				Split(items, result, RESULT_SEPARATOR);

				if (items.GetCount() > 0) {
					m_setup->Filter1ID = FilterIDFromName(items[0]);
					m_setupModified->Filter1ID = m_setup->Filter1ID;
				}

				if (items.GetCount() > 1) {
					m_setup->Filter2ID = FilterIDFromName(items[1]);
					m_setupModified->Filter2ID = m_setup->Filter2ID;
				}

				if (items.GetCount() > 2) {
					m_setup->Filter3ID = FilterIDFromName(items[2]);
					m_setupModified->Filter3ID = m_setup->Filter3ID;
				}
				processed = TRUE;
			}
			else if(command == _T("RS Aperture")){
				m_setup->ApertureID = ApertureIDFromName(result);
				m_setupModified->ApertureID = m_setup->ApertureID;
				processed = TRUE;
			}
			else if(command == _T("RS Mode")){
				m_setup->ModeID = ModeIDFromName(result);
				m_setupModified->ModeID = m_setup->ModeID;
				processed = TRUE;
			}
			else if(command == _T("RS RangeMode")) {
				m_setup->RangeModeID = RangeModeIDFromName(result);
				m_setupModified->RangeModeID = m_setup->RangeModeID;
				processed = TRUE;
			}
			else if(command == _T("RS Range")){
				m_setup->RangeID = RangeIDFromName(result);
				m_setupModified->RangeID = m_setup->RangeID;
				processed = TRUE;
			}
			else if(command == _T("RS Speed")){
				m_setup->SpeedID = SpeedIDFromName(result);
				m_setupModified->SpeedID = m_setup->SpeedID;
				processed = TRUE;
			}
			else if(command == _T("RS ExposureMode")){
				m_setup->ExposureModeID = ExposureModeIDFromName(result);
				m_setupModified->ExposureModeID = m_setup->ExposureModeID;
				processed = TRUE;
			}
			else if(command == _T("RS Exposure")){
				CStringArray items;
				Split(items, result, RESULT_SPACE);
				//OK:0:RS Exposure:1.000 msec
				if (items.GetCount() != 2) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				if (!_stscanf_s(items[0], _T("%f"), &m_setup->Exposure)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				m_setupModified->Exposure = m_setup->Exposure;
				processed = TRUE;
			}
			else if(command == _T("RS MaxAutoExposure")){
				CStringArray items;
				Split(items, result, RESULT_SPACE);
				//OK:0:RS MaxAutoExposure:1.000 msec
				if (items.GetCount() != 2) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				if (!_stscanf_s(items[0], _T("%f"), &m_setup->MaxAutoExposure)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				m_setupModified->MaxAutoExposure = m_setup->MaxAutoExposure;
				processed = TRUE;
			}
			else if(command == _T("RS SyncMode")){
				m_setup->SyncModeID = SyncModeIDFromName(result);
				m_setupModified->SyncModeID = m_setup->SyncModeID;
				processed = TRUE;
			}
			else if(command == _T("RS SyncFreq")){
				CStringArray items;
				Split(items, result, RESULT_SPACE);
				//OK:0:RS SyncFreq:60.00 Hz
				if (items.GetCount() != 2) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				if (!_stscanf_s(items[0], _T("%f"), &m_setup->SyncFreq)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				m_setupModified->SyncFreq = m_setup->SyncFreq;
				processed = TRUE;
			}
			else if(command == _T("RS ExposureX")){
				if(!_stscanf_s(result, _T("%d"), &m_setup->ExposureX)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				m_setupModified->ExposureX = m_setup->ExposureX;
				processed = TRUE;
			}
			else if(command == _T("RS MatrixMode")){
				m_setup->MatrixModeID = MatrixModeIDFromName(result);
				m_setupModified->MatrixModeID = m_setup->MatrixModeID;
				processed = TRUE;
			}
			else if(command == _T("RS UserCalibMode")){
				m_setup->UserCalibModeID = UserCalibModeIDFromName(result);
				m_setupModified->UserCalibModeID = m_setup->UserCalibModeID;
				processed = TRUE;
			}
			else if(command == _T("RS Matrix")){
				if (result = "None") {
					processed = TRUE;
					m_setup->MatrixID = -1;
				}
				else {
					if(!_stscanf_s(result, _T("%d"), &m_setup->MatrixID)) {
						
						AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
						
						AfxThrowUserException();
					}
				}
				m_setupModified->MatrixID = m_setup->MatrixID;

				processed = TRUE;
			}
			else if(command == _T("RS Match")) {
				if (result == "None") {
					processed = TRUE;
					m_setup->MatchID = -1;
				}
				else {
					if(!_stscanf_s(result, _T("%d"), &m_setup->MatchID)) {
						
						AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
						
						AfxThrowUserException();
					}
				}
				m_setupModified->MatchID = m_setup->MatchID;

				processed = TRUE;
			}
			else if(command == _T("RS SamplingRate")){
				/*CStringArray items;
				Split(items, result, RESULT_SPACE);
				//OK:0:RS SamplingRate:200.0 Hz
				if (items.GetCount() != 2) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}*/
				if (!_stscanf_s(result, _T("%f"), &m_setup->SamplingRate)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				m_setupModified->SamplingRate = m_setup->SamplingRate;
				processed = TRUE;
			}
			
			else if(command == _T("RS CMF")){
				
				if(!_stscanf_s(result, _T("%d"), &m_setup->CMF)) {
						
						AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
						
						AfxThrowUserException();
					}
				m_setupModified->CMF = m_setup->CMF;
				processed = TRUE;
			}
			//------------Setup Measurement------------
			else if(command == _T("SM Mode")){
				m_setup->ModeID = m_setupModified->ModeID;
				processed = TRUE;
			}
			else if(command == _T("SM Accessory")){
				m_setup->AccessoryID = m_setupModified->AccessoryID;
				processed = TRUE;
			}
			else if(command == _T("SM Filter1")){
				m_setup->Filter1ID = m_setupModified->Filter1ID;
				processed = TRUE;
			}
			else if(command == _T("SM Filter2")){
				m_setup->Filter2ID = m_setupModified->Filter2ID;
				processed = TRUE;
			}
			else if(command == _T("SM Filter3")){
				m_setup->Filter3ID = m_setupModified->Filter3ID;
				processed = TRUE;
			}
			else if(command == _T("SM Aperture")){
				m_setup->ApertureID = m_setupModified->ApertureID;
				processed = TRUE;
			}
			else if(command == _T("SM RangeMode")){
				m_setup->RangeModeID = m_setupModified->RangeModeID;
				processed = TRUE;
			}
			else if(command == _T("SM Range")){
				m_setup->RangeID = m_setupModified->RangeID;
				processed = TRUE;
			}
			else if(command == _T("SM Speed")){
				m_setup->SpeedID = m_setupModified->SpeedID;
				processed = TRUE;
			}
			else if(command == _T("SM ExposureMode")){
				m_setup->ExposureModeID = m_setupModified->ExposureModeID;
				processed = TRUE;
			}
			else if(command == _T("SM Exposure")){
				m_setup->Exposure = m_setupModified->Exposure;
				processed = TRUE;
			}
			else if(command == _T("SM MaxAutoExposure")){
				m_setup->MaxAutoExposure = m_setupModified->MaxAutoExposure;
				processed = TRUE;
			}
			else if(command == _T("SM SyncMode")){
				m_setup->SyncModeID = m_setupModified->SyncModeID;
				processed = TRUE;
			}
			else if(command == _T("SM SyncFreq")){
				m_setup->SyncFreq = m_setupModified->SyncFreq;
				processed = TRUE;
			}
			else if(command == _T("SM ExposureX")){
				m_setup->ExposureX = m_setupModified->ExposureX;
				processed = TRUE;
			}
			else if(command == _T("SM MatrixMode")){
				m_setup->MatrixModeID = m_setupModified->MatrixModeID;
				processed = TRUE;
			}
			else if(command == _T("SM UserCalibMode")){
				m_setup->UserCalibModeID = m_setupModified->UserCalibModeID;
				processed = TRUE;
			}
			else if(command == _T("SM Matrix")){
				m_setup->MatrixID = m_setupModified->MatrixID;
				processed = TRUE;
			}
			else if(command == _T("SM Match")){
				m_setup->MatchID = m_setupModified->MatchID;
				processed = TRUE;
			}
			else if(command == _T("SM SamplingRate")){
				m_setup->SamplingRate = m_setupModified->SamplingRate;
				processed = TRUE;
			}
			else if(command == _T("SM CMF")){
				m_setup->CMF = m_setupModified->CMF;
				processed = TRUE;
			}
			//------------Read Measurement------------
			else if(command == _T("RM ID")){
				m_reading->ID = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("ID"), (LPARAM) this);
			}
			else if(command == _T("RM Model")){
				m_reading->Model = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Model"), (LPARAM) this);
			}
			else if(command == _T("RM Time")){
				m_reading->Time = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Time"), (LPARAM) this);
			}
			else if(command == _T("RM Accessory")){
				m_reading->Accessory = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Accessory"), (LPARAM) this);
			}
			else if(command == _T("RM Filter")){
				m_reading->Filter = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Filter"), (LPARAM) this);
			}
			else if(command == _T("RM Aperture")){
				m_reading->Aperture = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Aperture"), (LPARAM) this);
			}
			else if(command == _T("RM Mode")){
				m_reading->Mode = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Mode"), (LPARAM) this);
			}
			else if(command == _T("RM ExposureMode")){
				m_reading->ExposureMode = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("ExposureMode"), (LPARAM) this);
			}
			else if(command == _T("RM Exposure")){
				m_reading->Exposure = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Exposure"), (LPARAM) this);
			}
			else if(command == _T("RM MaxAutoExposure")){
				m_reading->MaxAutoExposure = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("MaxAutoExposure"), (LPARAM) this);
			}
			else if(command == _T("RM RangeMode")){
				m_reading->RangeMode = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("RangeMode"), (LPARAM) this);
			}
			else if(command == _T("RM Range")){
				m_reading->Range = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Range"), (LPARAM) this);
			}
			else if(command == _T("RM Speed")){
				m_reading->Speed = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Speed"), (LPARAM) this);
			}
			else if(command == _T("RM SyncMode")){
				m_reading->SyncMode = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("SyncMode"), (LPARAM) this);
			}
			else if(command == _T("RM SyncFreq")){
				m_reading->SyncFreq = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("SyncFreq"), (LPARAM) this);
			}
			else if(command == _T("RM ExposureX")){
				m_reading->ExposureX = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("ExposureX"), (LPARAM) this);
			}
			else if(command == _T("RM MatrixMode")){
				m_reading->MatrixMode = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("MatrixMode"), (LPARAM) this);
			}
			else if(command == _T("RM UserCalibMode")){
				m_reading->UserCalibMode = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("UserCalibMode"), (LPARAM) this);
			}
			else if(command == _T("RM Matrix")){
				if (result == "None") {
					m_reading->MatrixID = "-1";
				}
				else {
					int matrixID = 0;
					if(!_stscanf_s(result, _T("%d"), &matrixID)) {
						
						AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
						
						AfxThrowUserException();
					}
					else
						m_reading->MatrixID = result;
					
				}
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Matrix"), (LPARAM) this);
			}
			else if(command == _T("RM Match")){
				if (result == "None") {
					m_reading->MatchID = "-1";
				}
				else {
					int matchID = 0;
					if(!_stscanf_s(result, _T("%d"), &matchID)) {
						
						AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
						
						AfxThrowUserException();
					}
					else
						m_reading->MatchID = result;
				}
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Match"), (LPARAM) this);
			}
			else if(command == _T("RM X")){
				m_reading->CIE[Observer_2Degree].X = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("X"), (LPARAM) this);
			}
			else if(command == _T("RM Y")){
				m_reading->CIE[Observer_2Degree].Y = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Y"), (LPARAM) this);
			}
			else if(command == _T("RM Z")){
				m_reading->CIE[Observer_2Degree].Z = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR)_T("Z"), (LPARAM) this);
			}
			else if(command == _T("RM XYZ")){
				m_reading->CIE[Observer_2Degree].XYZ = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("XYZ"), (LPARAM) this);
			}
			else if(command == _T("RM xy")){
				m_reading->CIE[Observer_2Degree].xy = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("xy"), (LPARAM) this);
			}
			else if(command == _T("RM uv")){
				m_reading->CIE[Observer_2Degree].uv = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("uv"), (LPARAM) this);
			}
			else if(command == _T("RM upvp")){
				m_reading->CIE[Observer_2Degree].upvp = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("upvp"), (LPARAM) this);
			}
			else if(command == _T("RM CCT")){
				m_reading->CIE[Observer_2Degree].CCT = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("CCT"), (LPARAM) this);
			}
			else if(command == _T("RM Yv")){
				m_reading->Yv = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Yv"), (LPARAM) this);
			}
			else if(command == _T("RM Radiometric")){
				m_reading->Radiometric = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Radiometric"), (LPARAM) this);
			}			
			else if(command == _T("RM CMF")){
				m_reading->CMF = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("CMF"), (LPARAM) this);
			}
			else if(command == _T("RM Warnings")){
				
				if(!_stscanf_s(result, _T("%d"), &itemLines)) {
					
					AfxMessageBox(CStringFormat(_T("Response for [%s] has invalid value in result:[%s]"), command, result));
					
					AfxThrowUserException();
				}
				lines += itemLines;
				m_reading->Warnings.SetSize(itemLines);
				if (m_responseQ.GetCount() > itemLines) {
					for(int index=1; index<=itemLines; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 2) { 
							
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));
							
							AfxThrowUserException();
						}

						if (!_stscanf_s(items[0], _T("%d"), &m_reading->Warnings[index - 1].Code)) {
							AfxMessageBox(_T("Id is not a valid number"));
							AfxThrowUserException();

						}
						m_reading->Warnings[index - 1].Description = items[1];
						m_reading->AllWarnings += itemResponse + _T("\r");

					}
					processed = TRUE;
					::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Warnings"), (LPARAM) this);


				}
			}
			else if(command == _T("RM X10")){
				m_reading->CIE[Observer_10Degree].X = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("X10"), (LPARAM) this);
			}
			else if(command == _T("RM Y10")){
				m_reading->CIE[Observer_10Degree].Y = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Y10"), (LPARAM) this);
			}
			else if(command == _T("RM Z10")){
				m_reading->CIE[Observer_10Degree].Z = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Z10"), (LPARAM) this);
			}
			else if(command == _T("RM XYZ10")){
				m_reading->CIE[Observer_10Degree].XYZ = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("XYZ10"), (LPARAM) this);
			}
			else if(command == _T("RM xy10")){
				m_reading->CIE[Observer_10Degree].xy = result;
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("xy10"), (LPARAM) this);
			}
			else if(command == _T("RM Spectrum")){
				CStringArray spectrumParams;
				Split(spectrumParams, result, RESULT_SEPARATOR);

				int numberOfPoints = 0;
				for(int paramIndex = 0; paramIndex<spectrumParams.GetCount(); paramIndex++) {

					switch(paramIndex) {
				case 0:
					if(!_stscanf_s(spectrumParams[paramIndex], _T("%f"), &m_reading->Spectrum.StartingWavelength)) {

						AfxMessageBox(_T("Data point is not a valid number"));
						AfxThrowUserException();
					}
					break;
				case 1:
					if(!_stscanf_s(spectrumParams[paramIndex], _T("%f"), &m_reading->Spectrum.EndingWavelength)) {

						AfxMessageBox(_T("Data point is not a valid number"));
						AfxThrowUserException();
					}
					break;
				case 2:
					if(!_stscanf_s(spectrumParams[paramIndex], _T("%f"), &m_reading->Spectrum.Delta)) {

						AfxMessageBox(_T("Data point is not a valid number"));
						AfxThrowUserException();
					}
					break;
				case 3:
					if(!_stscanf_s(spectrumParams[paramIndex], _T("%d"), &numberOfPoints)) {

						AfxMessageBox(_T("Data point is not a valid number"));
						AfxThrowUserException();
					}
					break;
				default:
					//Invalid command: Has extra params
					break;
					}
				}


				lines += numberOfPoints;
				m_reading->Spectrum.Data.SetSize(numberOfPoints);
				if (m_responseQ.GetCount() > numberOfPoints) {
					for(int index = 1; index <= numberOfPoints; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 1) { 
							
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));
							
							AfxThrowUserException();
						}

						if(!_stscanf_s(items[0], _T("%f"), &m_reading->Spectrum.Data[index - 1])) {

							AfxMessageBox(_T("Data point is not a valid number"));
							AfxThrowUserException();
						}

					}
					processed = TRUE;

					::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Spectrum"), (LPARAM)this);
				}
			}
			
			else if(command == _T("RM Temporal") || command == _T("RM TemporalY")){
				CStringArray temporalParams;
				Split(temporalParams, result, RESULT_SEPARATOR);

				int numberOfPoints = 0;
				for(int paramIndex = 0; paramIndex<temporalParams.GetCount(); paramIndex++) {

					switch(paramIndex) {
				case 0:
					if(!_stscanf_s(temporalParams[paramIndex], _T("%f"), &m_reading->Temporal.SamplingRate)) {

						AfxMessageBox(_T("Data point is not a valid number"));
						AfxThrowUserException();
					}
					break;
				case 1:
					if(!_stscanf_s(temporalParams[paramIndex], _T("%d"), &numberOfPoints)) {

						AfxMessageBox(_T("Data point is not a valid number"));
						AfxThrowUserException();
					}
					break;
				default:
					//Invalid command: Has extra params
					break;
					}
				}


				lines += numberOfPoints;
				m_reading->Temporal.Data.SetSize(numberOfPoints);
				if (m_responseQ.GetCount() > numberOfPoints) {
					for(int index = 1; index <= numberOfPoints; index++) {
						CString itemResponse = m_responseQ.GetAt((INT_PTR)index);
						itemResponse.Remove('\r');
						itemResponse.Remove('\n');
						CStringArray items;
						Split(items, itemResponse, RESULT_SEPARATOR);

						if (items.GetCount() != 1) { 
							
							AfxMessageBox(CStringFormat(_T("Item is missing values:[%s]"), itemResponse));
							
							AfxThrowUserException();
						}

						float point = 0;
						CString strPoint = items[0];
						if(!_stscanf_s(strPoint, _T("%g"), &point)) {

							AfxMessageBox(_T("Data point is not a valid number"));
							AfxThrowUserException();
						}
						m_reading->Temporal.Data.SetAt(index - 1,  point);
						//m_reading->Temporal.Data.Add(point);

					}
					processed = TRUE;

					::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_DATA_CHANGED, (WPARAM)(LPCTSTR) _T("Temporal"), (LPARAM)this);
				}
			}
			//------------Measure------------
			else if(command == _T("M")){
				processed = TRUE;
				::SendMessage(m_pOwner->m_hWnd, UWM_MEASUREMENT_CHANGED, 0, (LPARAM) this);
				DownloadReading();
			}
			else {
				//Unrecognized Command				
				AfxMessageBox(CStringFormat(_T("Unrecognized Command %s in the response %s"), command, response));				
				AfxThrowUserException();
			}
		}

		canProcess = FALSE;

		if(processed == TRUE) {
			TRACE(_T("Processed Command %s\n"), m_commandQ.GetHead());
			m_spentCommandQ.Push(m_commandQ.Pop());
			for(int line = 0; line< lines; line++) {
				TRACE(_T("Processed Response %s\n"), m_responseQ.GetHead());
				m_spentResponseQ.Push(m_responseQ.Pop());
			}
			if (m_responseQ.GetCount() > 0) {
				canProcess = TRUE;
			}
			OnCommandCompleted(command);
		}
	}
	while(canProcess);

	TRACE(_T("Finish Processing Response"));
}





