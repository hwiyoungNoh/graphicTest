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
// MainFrm.cpp : implementation of the CMainFrame class
//

#include "stdafx.h"
#include "RemoteCommunication.h"
#include "ConnectionSetup.h"
#include "SerialChannel.h"
#include "TcpChannel.h"
#include "CR100Setup.h"
#include "CR250Setup.h"

#include "MainFrm.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CMainFrame

IMPLEMENT_DYNAMIC(CMainFrame, CFrameWnd)

BEGIN_MESSAGE_MAP(CMainFrame, CFrameWnd)
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_WM_SETFOCUS()
	ON_COMMAND(ID_BUTTON_CAPTURE, OnCapture)	
	ON_COMMAND(ID_BUTTON_SETUP, OnSetup)	
#ifdef __USE_REGISTERED_WINDOWS_MESSAGES
	ON_REGISTERED_MESSAGE(UWM_MEASUREMENT_CHANGED, OnMessageMeasurementChanged)
	ON_REGISTERED_MESSAGE(UWM_MEASUREMENT_DATA_CHANGED, OnMessageMeasurementDataChanged)
	ON_REGISTERED_MESSAGE(UWM_DATA_ERROR, OnMessageDataError)
	ON_REGISTERED_MESSAGE(UWM_DATA_SENT, OnMessageDataSent)
	ON_REGISTERED_MESSAGE(UWM_DATA_RECEIVED, OnMessageReceived)
	ON_REGISTERED_MESSAGE(UWM_DATA_DEBUG, OnMessageDebug)
	ON_REGISTERED_MESSAGE(UWM_CONNECTED, OnMessageConnected)
	ON_REGISTERED_MESSAGE(UWM_DISCONNECTED, OnMessageDisconnected)
#else
	ON_MESSAGE(UWM_MEASUREMENT_CHANGED, OnMessageMeasurementChanged)
	ON_MESSAGE(UWM_MEASUREMENT_DATA_CHANGED, OnMessageMeasurementDataChanged)
	ON_MESSAGE(UWM_DATA_ERROR, OnMessageDataError)
	ON_MESSAGE(UWM_DATA_SENT, OnMessageDataSent)
	ON_MESSAGE(UWM_DATA_RECEIVED, OnMessageReceived)
	ON_MESSAGE(UWM_DATA_DEBUG, OnMessageDebug)
	ON_MESSAGE(UWM_CONNECTED, OnMessageConnected)
	ON_MESSAGE(UWM_DISCONNECTED, OnMessageDisconnected)
#endif
END_MESSAGE_MAP()

static UINT indicators[] =
{
	ID_SEPARATOR,           // status line indicator
	ID_INDICATOR_CAPS,
	ID_INDICATOR_NUM,
	ID_INDICATOR_SCRL,
};

// CMainFrame construction/destruction

CMainFrame::CMainFrame()
{
	// TODO: add member initialization code here
	m_channel = NULL;
	m_colorimeter.SetOwner(this);
	m_bInitSplitter = FALSE;
	m_pDataView = NULL;
	m_pResultsView = NULL;

	
}

CMainFrame::~CMainFrame()
{
	if(m_colorimeter.IsConnected())
		m_colorimeter.Disconnect();
	
	if(m_channel)
		delete m_channel;
	m_channel = NULL;

}

int CMainFrame::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CFrameWnd::OnCreate(lpCreateStruct) == -1)
		return -1;
	
	if (!m_wndToolBar.CreateEx(this, TBSTYLE_FLAT, WS_CHILD | WS_VISIBLE | CBRS_TOP
		| CBRS_GRIPPER | CBRS_TOOLTIPS | CBRS_FLYBY | CBRS_SIZE_DYNAMIC) ||
		!m_wndToolBar.LoadToolBar(IDR_MAINFRAME))
	{
		TRACE0("Failed to create toolbar\n");
		return -1;      // fail to create
	}

	if (!m_wndStatusBar.Create(this) ||
		!m_wndStatusBar.SetIndicators(indicators,
		  sizeof(indicators)/sizeof(UINT)))
	{
		TRACE0("Failed to create status bar\n");
		return -1;      // fail to create
	}

	// TODO: Delete these three lines if you don't want the toolbar to
	//  be dockable
	m_wndToolBar.EnableDocking(CBRS_ALIGN_ANY);
	EnableDocking(CBRS_ALIGN_ANY);
	DockControlBar(&m_wndToolBar);
	
    ModifyStyle(0, WS_CLIPCHILDREN);
	return 0;
}

int CMainFrame::OnCreateClient(LPCREATESTRUCT lpCreateStruct, CCreateContext* pContext)
{
	//calculate client size
	CRect cr;
	GetWindowRect( &cr );
	
	// Create the main splitter with 2 row and 1 columns
	if ( !m_mainSplitter.CreateStatic( this, 2, 1 ) )
	{
		MessageBox(_T("Error setting up m_mainSplitter"), _T("ERROR"), MB_OK | MB_ICONERROR );
		return FALSE;
	}


	// The views for each pane must be created 
	if ( !m_mainSplitter.CreateView( 0, 0, RUNTIME_CLASS(CDataView),
		CSize(cr.Width(), cr.Height()/2), pContext ) )
	{
		MessageBox(_T("Error setting up splitter view"), _T("ERROR"), MB_OK | MB_ICONERROR );
		return FALSE;
	}

	if ( !m_mainSplitter.CreateView( 1, 0, RUNTIME_CLASS(CResultsView),
		CSize(cr.Width()/2, cr.Height()/2), pContext ) )
	{
		MessageBox(_T("Error setting up splitter view"), _T("ERROR"), MB_OK | MB_ICONERROR );
		return FALSE;
	}

	m_pDataView = (CDataView*)m_mainSplitter.GetPane(0,0);
	m_pResultsView = (CResultsView*)m_mainSplitter.GetPane(1,0);

	//change flag to show splitter created
	m_bInitSplitter = true;

	
	m_pDataView->m_flickerSetupCtrl.SetColorimeter(&m_colorimeter);
	m_pDataView->m_flickerSetupCtrl.EnableSettings(FALSE);

	
	m_pDataView->m_responseTimeSetupCtrl.SetColorimeter(&m_colorimeter);
	m_pDataView->m_responseTimeSetupCtrl.EnableSettings(FALSE);

	
	m_pDataView->m_calculationSetupCtrl.SetColorimeter(&m_colorimeter);
	m_pDataView->m_calculationSetupCtrl.EnableSettings(FALSE);

	//return TRUE instead of the parent method since that would
	//not show our window
	return TRUE;
}

BOOL CMainFrame::PreCreateWindow(CREATESTRUCT& cs)
{
	if( !CFrameWnd::PreCreateWindow(cs) )
		return FALSE;
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	cs.dwExStyle &= ~WS_EX_CLIENTEDGE;
	cs.lpszClass = AfxRegisterWndClass(0);
	return TRUE;
}


void CMainFrame::OnSize(UINT nType, int cx, int cy) 
{
   CFrameWnd::OnSize(nType, cx, cy);
	
	/*CRect cr;
	GetWindowRect(&cr);

	if (m_bInitSplitter && nType != SIZE_MINIMIZED )
	{
		m_mainSplitter.SetRowInfo( 0, cx, 0 );
		m_mainSplitter.SetColumnInfo( 0, cr.Height() / 2, 50);
		m_mainSplitter.SetColumnInfo( 1, cr.Height() / 2, 50);
		
		m_mainSplitter.RecalcLayout();
	}	*/
}
// CMainFrame diagnostics

#ifdef _DEBUG
void CMainFrame::AssertValid() const
{
	CFrameWnd::AssertValid();
}

void CMainFrame::Dump(CDumpContext& dc) const
{
	CFrameWnd::Dump(dc);
}

#endif //_DEBUG


// CMainFrame message handlers

void CMainFrame::OnSetFocus(CWnd* /*pOldWnd*/)
{
	// forward focus to the view window
	//m_mainSplitter.SetFocus();

}

BOOL CMainFrame::OnCmdMsg(UINT nID, int nCode, void* pExtra, AFX_CMDHANDLERINFO* pHandlerInfo)
{
	// let the view have first crack at the command
	if (m_mainSplitter.OnCmdMsg(nID, nCode, pExtra, pHandlerInfo))
		return TRUE;

	// otherwise, do default handling
	return CFrameWnd::OnCmdMsg(nID, nCode, pExtra, pHandlerInfo);
}

LRESULT CMainFrame::OnMessageMeasurementChanged(WPARAM wParam, LPARAM lParam)
{
	m_pDataView->AddMeasurement();
	return 0;
}

LRESULT CMainFrame::OnMessageMeasurementDataChanged(WPARAM wParam, LPARAM lParam)
{
	CString dataType = (LPCTSTR)wParam;
	CString data;
	CCRColorimeter* meter = (CCRColorimeter*)lParam;
	if(!meter)
		return 0;
	if(dataType == "Spectrum")
	{		
		m_pDataView->AddSpectralData(_T("Start"), meter->Reading(_T("StartWavelength")));
		m_pDataView->AddSpectralData(_T("End"), meter->Reading(_T("EndWavelength")));
		m_pDataView->AddSpectralData(_T("Delta"), meter->Reading(_T("DeltaWavelength")));

		CArray<double,double>& spectrum =  meter->Spectrum();
		CString text;
		text.Format(_T("%d"), spectrum.GetCount());
		m_pDataView->AddSpectralData(_T("Points"), text);
		
		for(int i=0; i<spectrum.GetCount(); i++)
		{
			CString point;
			text.Format(_T("%d"), i+1);
			point.Format(_T("%.3e"), spectrum.GetAt(i));
			m_pDataView->AddSpectralData(text, point);
		}
	}
	else if(dataType == _T("Temporal"))
	{		
		m_pDataView->AddTemporalData(_T("Sampling Rate"), meter->Reading(_T("SamplingRate")));

		CArray<double,double>& temporal =  meter->Temporal();
        int points = (int)temporal.GetCount();
		CString text;
		text.Format(_T("%d"), temporal.GetCount());
		m_pDataView->AddTemporalData(_T("Points"), text);
		
		for(int i=0; i<temporal.GetCount(); i++)
		{
			CString point;
			text.Format(_T("%d"), i+1);
			point.Format(_T("%f"), temporal[i]);
			m_pDataView->AddTemporalData(text, point);
		}

		// Flicker
		flicker_t *flickerContext = 0;
        cs_flicker_data_t flickerData;
        cs_flicker_filter_t flickerFilterData;
        int32_t flickerResult = 0;
        double samplingRate = strtod(CStringA(meter->Reading(_T("SamplingRate"))).GetString(), NULL);

        if(points > 0)
        {
            flickerFilterData.filter_type = meter->FlickerFilterType();
            flickerFilterData.filter_family = meter->FlickerFilterFamily();
            flickerFilterData.order = meter->FlickerFilterOrder();
            flickerFilterData.frequency = meter->FlickerFilterFrequency();
            flickerFilterData.bandwidth = meter->FlickerFilterBandwidth();

            flickerResult = cs_flicker_filter(&flickerFilterData, samplingRate, temporal.GetData(), points, temporal.GetData());
            flickerContext = cs_flicker_create(samplingRate, temporal.GetData(), points, meter->FlickerMaxSearchFrequency());
            if (flickerContext != 0)
                flickerResult = cs_flicker_data(flickerContext, &flickerData);
            cs_flicker_free(flickerContext);
            flickerContext = 0;

        }

		CString strValue;	

		strValue.Format(_T("%f"), meter->FlickerMaxSearchFrequency());
		m_pDataView->AddFlickerData(_T("Max. Flicker Search Frequency"), strValue);
		m_pDataView->AddFlickerData(_T("Filter Type"), meter->FlickerFilterTypeName());

		strValue.Format(_T("%d"), meter->FlickerFilterOrder());
		m_pDataView->AddFlickerData(_T("Order"), strValue);
		
		strValue.Format(_T("%f"), meter->FlickerFilterFrequency());
		m_pDataView->AddFlickerData(_T("Frequency"), strValue);
		
		strValue.Format(_T("%f"), m_colorimeter.FlickerFilterBandwidth());
		m_pDataView->AddFlickerData(_T("Bandwidth"), strValue);


		strValue.Format(_T("%f"), flickerData.flicker_frequency);
		m_pDataView->AddFlickerData(_T("Flicker Frequency (Hz.)"), strValue);  //"Flicker Frequency (Hz.)"
		
		strValue.Format(_T("%f"), flickerData.flicker_level);
		m_pDataView->AddFlickerData(_T("JEITA Flicker (Weighted Level) dB."), strValue); //"JEITA Flicker (Weighted Level) dB."
		
		strValue.Format(_T("%f"), flickerData.percent_flicker);
		m_pDataView->AddFlickerData(_T("Percent Flicker (%)"), strValue); //"Percent Flicker (%)"
		
		strValue.Format(_T("%d"), flickerData.flicker_index);
		m_pDataView->AddFlickerData(_T("Flicker Index"), strValue); //"Flicker Index"
		
		strValue.Format(_T("%f"), flickerData.flicker_modulation_amplitude);
		m_pDataView->AddFlickerData(_T("FMA Flicker (Modulation) %"), strValue); //'"FMA Flicker (Modulation) %"
		
		// Response Time

		responsetime_t *responseTimeContext = 0;
        cs_responsetime_data_t responseTimeData;
        cs_responsetime_settings_t responseTimeSettings;
        int32_t responseTimeResult = 0;
        samplingRate = strtod(CStringA(meter->Reading(_T("SamplingRate"))).GetString(), NULL);

        if(points > 0)
        {
			uint8_t peaks = 0;
			uint8_t transition = 0;
			responseTimeSettings.mode = meter->ResponseTimeMode();
			responseTimeSettings.filter_type = meter->ResponseTimeFilterType();    // responsetime_filter_type
			responseTimeSettings.average = meter->ResponseTimeAverage();
			responseTimeSettings.clipping_enabled = meter->ResponseTimeClippingEnabled();
			responseTimeSettings.clipping_lo = meter->ResponseTimeClippingLowerLimit(); // %
			responseTimeSettings.clipping_hi = meter->ResponseTimeClippingUpperLimit(); // %
			responseTimeSettings.noiselevel = meter->ResponseTimeNoiseLevel();  // %
			responseTimeSettings.setupresponsezone_lo = meter->ResponseTimeStepResponseZoneLowerLimit();
			responseTimeSettings.setupresponsezone_hi = meter->ResponseTimeStepResponseZoneUpperLimit(); // %


            responseTimeContext = cs_responsetime_create(samplingRate, temporal.GetData(), points);
			//samplingRate = 40960.0;
            //responseTimeContext = cs_responsetime_create(samplingRate, InitSampleSineWave(), 1024);
            if (responseTimeContext != 0)
			{
				responseTimeResult = cs_responsetime_update(responseTimeContext, &responseTimeSettings);
				responseTimeResult = cs_responsetime_peaks(responseTimeContext, &peaks);

				if(peaks > 0)
				{
					transition = peaks/2 + 1; // get the middle one.
					responseTimeResult = cs_responsetime_data(responseTimeContext, transition, &responseTimeData);
				}

			}
            cs_responsetime_free(responseTimeContext);
            responseTimeContext = 0;

        }

		strValue = "";	

		strValue.Format(_T("%f"), responseTimeData.minimum);
		m_pDataView->AddResponseTimeData(_T("Minimum"), strValue);
		strValue.Format(_T("%f"), responseTimeData.maximum);
		m_pDataView->AddResponseTimeData(_T("Maximum"), strValue);
		
		strValue.Format(_T("%f"), responseTimeData.contrast);
		m_pDataView->AddResponseTimeData(_T("Contrast"), strValue);
		
		strValue.Format(_T("%f"), responseTimeData.valley1);
		m_pDataView->AddResponseTimeData(_T("Valley1"), strValue);
		strValue.Format(_T("%f"), responseTimeData.peak);
		m_pDataView->AddResponseTimeData(_T("Peak"), strValue);
		strValue.Format(_T("%f"), responseTimeData.valley2);
		m_pDataView->AddResponseTimeData(_T("Valley2"), strValue);
		
		strValue.Format(_T("%f"), responseTimeData.rise_time);
		m_pDataView->AddResponseTimeData(_T("Rise time (msecs.)"), strValue);
		
		strValue.Format(_T("%f"), responseTimeData.fall_time);
		m_pDataView->AddResponseTimeData(_T("Fall time (msecs.)"), strValue);  
		
		strValue.Format(_T("%f"), responseTimeData.response_time);
		m_pDataView->AddResponseTimeData(_T("Response time (msecs.)"), strValue);
		
				
	}
	else
	{	
		CString strValue;	
		strValue = "";
		if(dataType == _T("xy"))
		{
			cs_dominantwavelength_data_t dominantWavelengthData;
            memset(&dominantWavelengthData, 0, sizeof(cs_dominantwavelength_data_t));

			double x = 0.0;
			double y = 0.0;
			if(_stscanf_s(data, _T("%f,%f"), &x, &y)) {	
		
                dominantWavelengthData.x = x;
                dominantWavelengthData.y = y;
            }
            uint32_t dominantWavelengthResult = 0;
			        
            dominantWavelengthData.xWhite = 0.3127F;
            dominantWavelengthData.yWhite = 0.3290F;
		
            dominantWavelengthResult = cs_dominantwavelength(&dominantWavelengthData);

			if(dominantWavelengthResult == 0)
            {
                if(dominantWavelengthData.isComplimentary)
					strValue.Format(_T("%f[C]"), dominantWavelengthData.dominantWavelength);
                else
                   strValue.Format(_T("%f"), dominantWavelengthData.dominantWavelength);
				
				m_pDataView->AddCalculationData(_T("Dominant Wavelength (nm.)"), strValue);
				
                strValue.Format(_T("%f"), dominantWavelengthData.purity);
				m_pDataView->AddCalculationData(_T("Purity (%)"), strValue);
            }
            else
            {
				m_pDataView->AddCalculationData(_T("Dominant Wavelength (nm.)"), "ERR");
				m_pDataView->AddCalculationData(_T("Purity (%)"), "ERR");
            }
		}
		data = meter->Reading(dataType);
		m_pDataView->AddMeasurementData(dataType, data);
	}
	return 0;
}

LRESULT CMainFrame::OnMessageDataError(WPARAM wParam, LPARAM lParam)
{
	CString buffer = (LPCTSTR)wParam;
	if(m_pResultsView)
		m_pResultsView->InsertText(buffer, RGB(196, 0, 0), true, false);
	return 0;
}

LRESULT CMainFrame::OnMessageDataSent(WPARAM wParam, LPARAM lParam)
{
	CString buffer = (LPCTSTR)wParam;
	if(m_pResultsView)
		m_pResultsView->InsertText(buffer, RGB(0, 0, 196), true, false);
	return 0;
}

LRESULT CMainFrame::OnMessageReceived(WPARAM wParam, LPARAM lParam)
{
	CString buffer = (LPCTSTR)wParam;
	if(m_pResultsView)
		m_pResultsView->InsertText(buffer, RGB(0, 196, 0), true, false);
	return 0;
}

LRESULT CMainFrame::OnMessageDebug(WPARAM wParam, LPARAM lParam)
{
	CString buffer = (LPCTSTR)wParam;
	if(m_pResultsView)
		m_pResultsView->InsertText(buffer, RGB(0, 0, 0), true, false);
	return 0;
}

LRESULT CMainFrame::OnMessageConnected(WPARAM wParam, LPARAM lParam)
{
	m_pDataView->EnableWindow(TRUE);
	m_pDataView->m_flickerSetupCtrl.EnableSettings(TRUE);
	m_pDataView->m_responseTimeSetupCtrl.EnableSettings(TRUE);
	m_pDataView->m_calculationSetupCtrl.EnableSettings(TRUE);
	return 0;
}

LRESULT CMainFrame::OnMessageDisconnected(WPARAM wParam, LPARAM lParam)
{	
	m_pDataView->EnableWindow(FALSE);
	m_pDataView->m_flickerSetupCtrl.EnableSettings(FALSE);
	m_pDataView->m_responseTimeSetupCtrl.EnableSettings(FALSE);
	m_pDataView->m_calculationSetupCtrl.EnableSettings(FALSE);

	return 0;
}

void CMainFrame::OnCapture()
{
	if(m_colorimeter.IsConnected())
	{
		m_colorimeter.Capture();	
	}
}

void CMainFrame::OnSetup()
{
	if(!m_colorimeter.IsConnected())
	{
		CConnectionSetup setup;
		if (setup.DoModal() == IDOK) 
		{
			if (m_channel != NULL) 
			{
				if (m_channel->IsOpen()) 
				{
					m_channel->Close();
				}
				delete m_channel;
				m_channel = NULL;
			}
			//if (setup.m_radioButtonSerial.Checked)
			{
				CSerialChannel* serialChannel = new CSerialChannel(); 
				serialChannel->SetPortName(setup.ChannelName());
				m_channel = serialChannel;
			}
			//else
			//{
			//	CTcpChannel* tcpChannel = new CTcpChannel(); 
			//	tcpChannel->SetHostName(setup.ChannelName());
			//	m_channel = tcpChannel;
			//}

			m_channel->Open();
			m_colorimeter.SetChannel(*m_channel);
			m_colorimeter.Connect();
		}

	}
	else
	{
		if (m_colorimeter.Model() == "CR-250")
		{
			CCR250Setup setup;
			setup.SetColorimeter(&m_colorimeter);  
			setup.DoModal();
		}
		else
		{
			CCR100Setup setup;
			setup.SetColorimeter(&m_colorimeter);
			setup.DoModal();
		}
	}


}

LRESULT CMainFrame::WindowProc(UINT message, WPARAM wParam, LPARAM lParam)
{
	return CFrameWnd::WindowProc(message, wParam, lParam);
}

double* CMainFrame::InitSampleSineWave()
{
	static double TemporalData[] = { 
	50,
    50.9203,
    51.8404,
    52.7598,
    53.6782,
    54.5954,
    55.5111,
    56.4249,
    57.3365,
    58.2457,
    59.152,
    60.0552,
    60.9551,
    61.8512,
    62.7433,
    63.6311,
    64.5142,
    65.3925,
    66.2655,
    67.133,
    67.9948,
    68.8504,
    69.6996,
    70.5422,
    71.3778,
    72.2061,
    73.0269,
    73.84,
    74.6449,
    75.4415,
    76.2295,
    77.0086,
    77.7785,
    78.539,
    79.2899,
    80.0308,
    80.7616,
    81.4819,
    82.1916,
    82.8903,
    83.5779,
    84.2542,
    84.9188,
    85.5716,
    86.2124,
    86.8408,
    87.4568,
    88.0601,
    88.6505,
    89.2278,
    89.7918,
    90.3424,
    90.8792,
    91.4023,
    91.9112,
    92.406,
    92.8864,
    93.3523,
    93.8035,
    94.2399,
    94.6612,
    95.0674,
    95.4584,
    95.834,
    96.194,
    96.5383,
    96.867,
    97.1797,
    97.4764,
    97.7571,
    98.0215,
    98.2697,
    98.5016,
    98.717,
    98.9159,
    99.0982,
    99.2639,
    99.4129,
    99.5451,
    99.6606,
    99.7592,
    99.841,
    99.9059,
    99.9539,
    99.9849,
    99.9991,
    99.9962,
    99.9765,
    99.9398,
    99.8862,
    99.8156,
    99.7282,
    99.624,
    99.5029,
    99.3651,
    99.2105,
    99.0393,
    98.8514,
    98.647,
    98.4261,
    98.1888,
    97.9352,
    97.6653,
    97.3793,
    97.0772,
    96.7592,
    96.4253,
    96.0757,
    95.7105,
    95.3298,
    94.9337,
    94.5224,
    94.0961,
    93.6547,
    93.1986,
    92.7279,
    92.2427,
    91.7431,
    91.2295,
    90.7018,
    90.1604,
    89.6053,
    89.0369,
    88.4552,
    87.8604,
    87.2529,
    86.6327,
    86.0001,
    85.3553,
    84.6986,
    84.03,
    83.35,
    82.6586,
    81.9562,
    81.243,
    80.5191,
    79.785,
    79.0407,
    78.2866,
    77.5229,
    76.7499,
    75.9678,
    75.1769,
    74.3775,
    73.5698,
    72.7542,
    71.9308,
    71.1,
    70.2621,
    69.4173,
    68.5659,
    67.7082,
    66.8445,
    65.9751,
    65.1003,
    64.2204,
    63.3356,
    62.4464,
    61.5529,
    60.6555,
    59.7545,
    58.8502,
    57.9429,
    57.0329,
    56.1205,
    55.2061,
    54.2899,
    53.3722,
    52.4534,
    51.5337,
    50.6136,
    49.6932,
    48.7729,
    47.8531,
    46.934,
    46.0159,
    45.0991,
    44.1841,
    43.271,
    42.3601,
    41.4519,
    40.5466,
    39.6444,
    38.7458,
    37.851,
    36.9603,
    36.074,
    35.1925,
    34.3159,
    33.4447,
    32.5791,
    31.7194,
    30.8658,
    30.0188,
    29.1785,
    28.3453,
    27.5194,
    26.7012,
    25.8908,
    25.0886,
    24.2949,
    23.5098,
    22.7338,
    21.9669,
    21.2096,
    20.462,
    19.7244,
    18.9971,
    18.2803,
    17.5743,
    16.8792,
    16.1954,
    15.523,
    14.8623,
    14.2135,
    13.5768,
    12.9524,
    12.3407,
    11.7416,
    11.1556,
    10.5827,
    10.0231,
    9.47714,
    8.94487,
    8.42652,
    7.92225,
    7.43224,
    6.95665,
    6.49565,
    6.04939,
    5.61802,
    5.20169,
    4.80054,
    4.4147,
    4.04431,
    3.68949,
    3.35036,
    3.02704,
    2.71963,
    2.42825,
    2.15298,
    1.89393,
    1.65118,
    1.42481,
    1.21489,
    1.02151,
    0.844726,
    0.684595,
    0.541175,
    0.414512,
    0.304651,
    0.211629,
    0.135477,
    0.076221,
    0.0338808,
    0.00847091,
    0,
    0.00847091,
    0.0338808,
    0.076221,
    0.135477,
    0.211629,
    0.304651,
    0.414512,
    0.541175,
    0.684595,
    0.844726,
    1.02151,
    1.21489,
    1.42481,
    1.65118,
    1.89393,
    2.15298,
    2.42825,
    2.71963,
    3.02704,
    3.35036,
    3.68949,
    4.04431,
    4.4147,
    4.80054,
    5.20169,
    5.61802,
    6.04939,
    6.49565,
    6.95665,
    7.43224,
    7.92225,
    8.42652,
    8.94487,
    9.47714,
    10.0231,
    10.5827,
    11.1556,
    11.7416,
    12.3407,
    12.9524,
    13.5768,
    14.2135,
    14.8623,
    15.523,
    16.1954,
    16.8792,
    17.5743,
    18.2803,
    18.9971,
    19.7244,
    20.462,
    21.2096,
    21.9669,
    22.7338,
    23.5098,
    24.2949,
    25.0886,
    25.8908,
    26.7012,
    27.5194,
    28.3453,
    29.1785,
    30.0188,
    30.8658,
    31.7194,
    32.5791,
    33.4447,
    34.3159,
    35.1925,
    36.074,
    36.9603,
    37.851,
    38.7458,
    39.6444,
    40.5466,
    41.4519,
    42.3601,
    43.271,
    44.1841,
    45.0991,
    46.0159,
    46.934,
    47.8531,
    48.7729,
    49.6932,
    50.6136,
    51.5337,
    52.4534,
    53.3722,
    54.2899,
    55.2061,
    56.1205,
    57.0329,
    57.9429,
    58.8502,
    59.7545,
    60.6555,
    61.5529,
    62.4464,
    63.3356,
    64.2204,
    65.1003,
    65.9751,
    66.8445,
    67.7082,
    68.5659,
    69.4173,
    70.2621,
    71.1,
    71.9308,
    72.7542,
    73.5698,
    74.3775,
    75.1769,
    75.9678,
    76.7499,
    77.5229,
    78.2866,
    79.0407,
    79.785,
    80.5191,
    81.243,
    81.9562,
    82.6586,
    83.35,
    84.03,
    84.6986,
    85.3553,
    86.0001,
    86.6327,
    87.2529,
    87.8604,
    88.4552,
    89.0369,
    89.6053,
    90.1604,
    90.7018,
    91.2295,
    91.7431,
    92.2427,
    92.7279,
    93.1986,
    93.6547,
    94.0961,
    94.5224,
    94.9337,
    95.3298,
    95.7105,
    96.0757,
    96.4253,
    96.7592,
    97.0772,
    97.3793,
    97.6653,
    97.9352,
    98.1888,
    98.4261,
    98.647,
    98.8514,
    99.0393,
    99.2105,
    99.3651,
    99.5029,
    99.624,
    99.7282,
    99.8156,
    99.8862,
    99.9398,
    99.9765,
    99.9962,
    99.9991,
    99.9849,
    99.9539,
    99.9059,
    99.841,
    99.7592,
    99.6606,
    99.5451,
    99.4129,
    99.2639,
    99.0982,
    98.9159,
    98.717,
    98.5016,
    98.2697,
    98.0215,
    97.7571,
    97.4764,
    97.1797,
    96.867,
    96.5383,
    96.194,
    95.834,
    95.4584,
    95.0674,
    94.6612,
    94.2399,
    93.8035,
    93.3523,
    92.8864,
    92.406,
    91.9112,
    91.4023,
    90.8792,
    90.3424,
    89.7918,
    89.2278,
    88.6505,
    88.0601,
    87.4568,
    86.8408,
    86.2124,
    85.5716,
    84.9188,
    84.2542,
    83.5779,
    82.8903,
    82.1916,
    81.4819,
    80.7616,
    80.0308,
    79.2899,
    78.539,
    77.7785,
    77.0086,
    76.2295,
    75.4415,
    74.6449,
    73.84,
    73.0269,
    72.2061,
    71.3778,
    70.5422,
    69.6996,
    68.8504,
    67.9948,
    67.133,
    66.2655,
    65.3925,
    64.5142,
    63.6311,
    62.7433,
    61.8512,
    60.9551,
    60.0552,
    59.152,
    58.2457,
    57.3365,
    56.4249,
    55.5111,
    54.5954,
    53.6782,
    52.7598,
    51.8404,
    50.9203,
    50,
    49.0797,
    48.1596,
    47.2402,
    46.3218,
    45.4046,
    44.4889,
    43.5751,
    42.6635,
    41.7543,
    40.848,
    39.9448,
    39.0449,
    38.1488,
    37.2567,
    36.3689,
    35.4858,
    34.6075,
    33.7345,
    32.867,
    32.0052,
    31.1496,
    30.3004,
    29.4578,
    28.6222,
    27.7939,
    26.9731,
    26.16,
    25.3551,
    24.5585,
    23.7705,
    22.9914,
    22.2215,
    21.461,
    20.7101,
    19.9692,
    19.2384,
    18.5181,
    17.8084,
    17.1097,
    16.4221,
    15.7458,
    15.0812,
    14.4284,
    13.7876,
    13.1592,
    12.5432,
    11.9399,
    11.3495,
    10.7722,
    10.2082,
    9.65762,
    9.12076,
    8.59775,
    8.08876,
    7.59398,
    7.11357,
    6.64769,
    6.1965,
    5.76015,
    5.33878,
    4.93256,
    4.5416,
    4.16605,
    3.80602,
    3.46165,
    3.13305,
    2.82033,
    2.52359,
    2.24294,
    1.97847,
    1.73028,
    1.49844,
    1.28303,
    1.08413,
    0.901807,
    0.736118,
    0.587122,
    0.454868,
    0.339403,
    0.240764,
    0.158985,
    0.0940944,
    0.0461136,
    0.0150591,
    0.000941236,
    0.00376491,
    0.0235291,
    0.0602272,
    0.113847,
    0.184369,
    0.271771,
    0.376023,
    0.497089,
    0.634929,
    0.789495,
    0.960736,
    1.14859,
    1.353,
    1.5739,
    1.8112,
    2.06483,
    2.3347,
    2.62072,
    2.9228,
    3.24082,
    3.5747,
    3.9243,
    4.28951,
    4.67021,
    5.06628,
    5.47756,
    5.90394,
    6.34525,
    6.80136,
    7.2721,
    7.75732,
    8.25686,
    8.77053,
    9.29818,
    9.83962,
    10.3947,
    10.9631,
    11.5448,
    12.1396,
    12.7471,
    13.3673,
    13.9999,
    14.6447,
    15.3014,
    15.97,
    16.65,
    17.3414,
    18.0438,
    18.757,
    19.4809,
    20.215,
    20.9593,
    21.7134,
    22.4771,
    23.2501,
    24.0322,
    24.8231,
    25.6225,
    26.4302,
    27.2458,
    28.0692,
    28.9,
    29.7379,
    30.5827,
    31.4341,
    32.2918,
    33.1555,
    34.0249,
    34.8997,
    35.7796,
    36.6644,
    37.5536,
    38.4471,
    39.3445,
    40.2455,
    41.1498,
    42.0571,
    42.9671,
    43.8795,
    44.7939,
    45.7101,
    46.6278,
    47.5466,
    48.4663,
    49.3864,
    50.3068,
    51.2271,
    52.1469,
    53.066,
    53.9841,
    54.9009,
    55.8159,
    56.729,
    57.6399,
    58.5481,
    59.4534,
    60.3556,
    61.2542,
    62.149,
    63.0397,
    63.926,
    64.8075,
    65.6841,
    66.5553,
    67.4209,
    68.2806,
    69.1342,
    69.9812,
    70.8215,
    71.6547,
    72.4806,
    73.2988,
    74.1092,
    74.9114,
    75.7051,
    76.4902,
    77.2662,
    78.0331,
    78.7904,
    79.538,
    80.2756,
    81.0029,
    81.7197,
    82.4257,
    83.1208,
    83.8046,
    84.477,
    85.1377,
    85.7865,
    86.4232,
    87.0476,
    87.6593,
    88.2584,
    88.8444,
    89.4173,
    89.9769,
    90.5229,
    91.0551,
    91.5735,
    92.0777,
    92.5678,
    93.0433,
    93.5043,
    93.9506,
    94.382,
    94.7983,
    95.1995,
    95.5853,
    95.9557,
    96.3105,
    96.6496,
    96.973,
    97.2804,
    97.5718,
    97.847,
    98.1061,
    98.3488,
    98.5752,
    98.7851,
    98.9785,
    99.1553,
    99.3154,
    99.4588,
    99.5855,
    99.6953,
    99.7884,
    99.8645,
    99.9238,
    99.9661,
    99.9915,
    100,
    99.9915,
    99.9661,
    99.9238,
    99.8645,
    99.7884,
    99.6953,
    99.5855,
    99.4588,
    99.3154,
    99.1553,
    98.9785,
    98.7851,
    98.5752,
    98.3488,
    98.1061,
    97.847,
    97.5718,
    97.2804,
    96.973,
    96.6496,
    96.3105,
    95.9557,
    95.5853,
    95.1995,
    94.7983,
    94.382,
    93.9506,
    93.5043,
    93.0433,
    92.5678,
    92.0777,
    91.5735,
    91.0551,
    90.5229,
    89.9769,
    89.4173,
    88.8444,
    88.2584,
    87.6593,
    87.0476,
    86.4232,
    85.7865,
    85.1377,
    84.477,
    83.8046,
    83.1208,
    82.4257,
    81.7197,
    81.0029,
    80.2756,
    79.538,
    78.7904,
    78.0331,
    77.2662,
    76.4902,
    75.7051,
    74.9114,
    74.1092,
    73.2988,
    72.4806,
    71.6547,
    70.8215,
    69.9812,
    69.1342,
    68.2806,
    67.4209,
    66.5553,
    65.6841,
    64.8075,
    63.926,
    63.0397,
    62.149,
    61.2542,
    60.3556,
    59.4534,
    58.5481,
    57.6399,
    56.729,
    55.8159,
    54.9009,
    53.9841,
    53.066,
    52.1469,
    51.2271,
    50.3068,
    49.3864,
    48.4663,
    47.5466,
    46.6278,
    45.7101,
    44.7939,
    43.8795,
    42.9671,
    42.0571,
    41.1498,
    40.2455,
    39.3445,
    38.4471,
    37.5536,
    36.6644,
    35.7796,
    34.8997,
    34.0249,
    33.1555,
    32.2918,
    31.4341,
    30.5827,
    29.7379,
    28.9,
    28.0692,
    27.2458,
    26.4302,
    25.6225,
    24.8231,
    24.0322,
    23.2501,
    22.4771,
    21.7134,
    20.9593,
    20.215,
    19.4809,
    18.757,
    18.0438,
    17.3414,
    16.65,
    15.97,
    15.3014,
    14.6447,
    13.9999,
    13.3673,
    12.7471,
    12.1396,
    11.5448,
    10.9631,
    10.3947,
    9.83962,
    9.29818,
    8.77053,
    8.25686,
    7.75732,
    7.2721,
    6.80136,
    6.34525,
    5.90394,
    5.47756,
    5.06628,
    4.67021,
    4.28951,
    3.9243,
    3.5747,
    3.24082,
    2.9228,
    2.62072,
    2.3347,
    2.06483,
    1.8112,
    1.5739,
    1.353,
    1.14859,
    0.960736,
    0.789495,
    0.634929,
    0.497089,
    0.376023,
    0.271771,
    0.184369,
    0.113847,
    0.0602272,
    0.0235291,
    0.00376491,
    0.000941236,
    0.0150591,
    0.0461136,
    0.0940944,
    0.158985,
    0.240764,
    0.339403,
    0.454868,
    0.587122,
    0.736118,
    0.901807,
    1.08413,
    1.28303,
    1.49844,
    1.73028,
    1.97847,
    2.24294,
    2.52359,
    2.82033,
    3.13305,
    3.46165,
    3.80602,
    4.16605,
    4.5416,
    4.93256,
    5.33878,
    5.76015,
    6.1965,
    6.64769,
    7.11357,
    7.59398,
    8.08876,
    8.59775,
    9.12076,
    9.65762,
    10.2082,
    10.7722,
    11.3495,
    11.9399,
    12.5432,
    13.1592,
    13.7876,
    14.4284,
    15.0812,
    15.7458,
    16.4221,
    17.1097,
    17.8084,
    18.5181,
    19.2384,
    19.9692,
    20.7101,
    21.461,
    22.2215,
    22.9914,
    23.7705,
    24.5585,
    25.3551,
    26.16,
    26.9731,
    27.7939,
    28.6222,
    29.4578,
    30.3004,
    31.1496,
    32.0052,
    32.867,
    33.7345,
    34.6075,
    35.4858,
    36.3689,
    37.2567,
    38.1488,
    39.0449,
    39.9448,
    40.848,
    41.7543,
    42.6635,
    43.5751,
    44.4889,
    45.4046,
    46.3218,
    47.2402,
    48.1596,
    49.0797 }
;	
return &TemporalData[0];
}		 
		 
		 
		 
		 
		 
		 
		 
		 
		 
		 
		 
		 
		 
		 
		 
		 
		 
		 