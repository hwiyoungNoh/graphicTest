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
// CR100Setup.cpp : implementation file
//

#include "stdafx.h"
#include "RemoteCommunication.h"
#include "CR100Setup.h"
#include <CRColorimeter.h>


// CCR100Setup dialog

IMPLEMENT_DYNAMIC(CCR100Setup, CDialogEx)

CCR100Setup::CCR100Setup(CWnd* pParent /*=NULL*/)
	: CDialogEx(CCR100Setup::IDD, pParent)
	, m_mode(0)
	, m_aperture(0)
	, m_accessory(0)
	, m_rangeMode(0)
	, m_exposureMode(0)
	, m_exposure(0)
	, m_maxAutoExposure(0)
	, m_minExposure(0)
	, m_maxExposure(0)
	, m_syncMode(0)
	, m_syncFreq(0)
	, m_minSyncFreq(0)
	, m_maxSyncFreq(0)
	, m_exposureX(0)
	, m_minExposureX(0)
	, m_maxExposureX(0)
	, m_range(0)
	, m_userCalibMode(0)
	, m_matrix(0)
	, m_match(0)
	, m_version(_T(""))
	, m_samplingRate(0)
	, m_minSamplingRate(0)
	, m_maxSamplingRate(0)
{
	m_colorimeter = NULL;
}


CCR100Setup::~CCR100Setup()
{
}

void CCR100Setup::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_BUTTON_DISCONNECT, m_disconnectButton);
	DDX_Control(pDX, IDOK, m_okButton);
	DDX_Control(pDX, IDC_STATIC_VERSION, m_versionLabel);
	DDX_Control(pDX, IDC_COMBO_MODE, m_modeComboBox);
	DDX_Control(pDX, IDC_COMBO_APERTURE, m_apertureComboBox);
	DDX_Control(pDX, IDC_COMBO_ACCESSORY, m_accessoryComboBox);
	DDX_Control(pDX, IDC_LIST_FILTERS, m_filtersCheckedListBox);
	DDX_Control(pDX, IDC_COMBO_RANGEMODE, m_rangeModeComboBox);
	DDX_Control(pDX, IDC_COMBO_RANGE, m_rangeComboBox);
	DDX_Control(pDX, IDC_COMBO_EXPOSUREMODE, m_exposureModeComboBox);
	DDX_Control(pDX, IDC_EDIT_EXPOSURE, m_exposureTextBox);
	DDX_Control(pDX, IDC_EDIT_MAXAUTOEXPOSURE, m_maxAutoExposureTextBox);
	DDX_Control(pDX, IDC_COMBO_SYNCMODE, m_syncModeComboBox);
	DDX_Control(pDX, IDC_EDIT_SYNCFREQ, m_syncFreqTextBox);
	DDX_Control(pDX, IDC_EDIT_EXPOSUREX, m_exposureXTextBox);
	DDX_Control(pDX, IDC_COMBO_USERCALIBMODE, m_userCalibModeComboBox);
	DDX_Control(pDX, IDC_COMBO_MATRIX, m_matrixComboBox);
	DDX_Control(pDX, IDC_COMBO_MATCH, m_matchComboBox);
	DDX_CBIndex(pDX, IDC_COMBO_MODE, m_mode);
	DDX_CBIndex(pDX, IDC_COMBO_APERTURE, m_aperture);
	DDX_CBIndex(pDX, IDC_COMBO_ACCESSORY, m_accessory);
	DDX_CBIndex(pDX, IDC_COMBO_RANGEMODE, m_rangeMode);
	DDX_CBIndex(pDX, IDC_COMBO_EXPOSUREMODE, m_exposureMode);
	DDX_Text(pDX, IDC_EDIT_EXPOSURE, m_exposure);
	DDX_Text(pDX, IDC_EDIT_MAXAUTOEXPOSURE, m_maxAutoExposure);
	DDX_CBIndex(pDX, IDC_COMBO_SYNCMODE, m_syncMode);
	DDX_Text(pDX, IDC_EDIT_SYNCFREQ, m_syncFreq);
	DDX_Text(pDX, IDC_EDIT_EXPOSUREX, m_exposureX);
	DDX_CBIndex(pDX, IDC_COMBO_RANGE, m_range);
	DDX_CBIndex(pDX, IDC_COMBO_USERCALIBMODE, m_userCalibMode);
	DDX_CBIndex(pDX, IDC_COMBO_MATRIX, m_matrix);
	DDX_CBIndex(pDX, IDC_COMBO_MATCH, m_match);
	DDX_Text(pDX, IDC_STATIC_VERSION, m_version);
	DDX_Control(pDX, IDC_SPIN_EXPOSREX, m_exposureSpinButton);
	DDX_Control(pDX, IDC_BUTTON_MATRIX, m_matrixEditButton);
	DDX_Control(pDX, IDC_BUTTON_MATCH, m_matchEditButton);
	DDX_Control(pDX, IDC_EDIT_SAMPLINGRATE, m_samplingRateTextBox);
	DDX_Text(pDX, IDC_EDIT_SAMPLINGRATE, m_samplingRate);


	//customized data verification
	if(pDX->m_bSaveAndValidate)
	{
		BOOL result = FALSE;
		result = validateFilters();
		if (result) result = validateSyncFreq();
		if (result) result = validateExposure();
		if (result) result = validateMaxAutoExposure();
		if (result) result = validateExposureX();
		if (result) result = validateSamplingRate();
		if (!result)
			pDX->Fail();   
	}

}


BEGIN_MESSAGE_MAP(CCR100Setup, CDialogEx)
	ON_BN_CLICKED(IDC_BUTTON_DISCONNECT, &CCR100Setup::OnBnClickedButtonDisconnect)
	ON_CBN_SELCHANGE(IDC_COMBO_ACCESSORY, &CCR100Setup::OnCbnSelchangeComboAccessory)
END_MESSAGE_MAP()


// CCR100Setup message handlers
BOOL CCR100Setup::OnInitDialog()
{
	CDialogEx::OnInitDialog();
	// Extra initialization here

	m_filtersCheckedListBox.SetExtendedStyle(m_filtersCheckedListBox.GetStyle()|LVS_EX_CHECKBOXES|LVS_EX_FULLROWSELECT&~LVS_EX_GRIDLINES);
	m_filtersCheckedListBox.InsertColumn(0, _T("Filter") ,LVCFMT_LEFT, 150);
	m_filtersCheckedListBox.SetColumnWidth(0, LVSCW_AUTOSIZE_USEHEADER);
	LoadSetup();

	UpdateData(FALSE);

	return TRUE;
}


CCRColorimeter* CCR100Setup::Colorimeter()
{
	return m_colorimeter;
}

void CCR100Setup::SetColorimeter(CCRColorimeter* colorimeter)
{
	m_colorimeter = colorimeter;
}

void CCR100Setup::LoadSetup()
{
	if(m_colorimeter)
	{
		CString strText;
		strText.Format(_T("Colorimeter Setup %s[%s]"), m_colorimeter->Model(), m_colorimeter->ID());
		SetWindowText(strText);

		m_disconnectButton.EnableWindow(m_colorimeter->IsConnected());
		m_okButton.EnableWindow(m_colorimeter->IsConnected());

		strText.Format(_T("Version: %s"), m_colorimeter->Firmware());
		m_version = strText;
		m_versionLabel.SetWindowText(strText);

		for(int i=0; i< m_colorimeter->ModeCount(); i++){
			m_modeComboBox.AddString(m_colorimeter->ModeName(i));
		}
		m_mode = m_colorimeter->Mode();
		m_modeComboBox.SetCurSel(m_colorimeter->Mode());


		for(int i=0; i< m_colorimeter->ApertureCount(); i++){
			m_apertureComboBox.AddString(m_colorimeter->ApertureName(i));
		}
		m_aperture = m_colorimeter->Aperture();
		m_apertureComboBox.SetCurSel(m_aperture);

		for(int i=0; i< m_colorimeter->AccessoryCount(); i++){
			m_accessoryComboBox.AddString(m_colorimeter->AccessoryName(i));
		}
		m_accessory = m_colorimeter->Accessory();
		m_accessoryComboBox.SetCurSel(m_accessory);

		for(int i=0; i< m_colorimeter->FilterCount(); i++){
			m_filtersCheckedListBox.InsertItem(i, m_colorimeter->FilterName(i));
		}
				   
		for(int i=0; i< m_colorimeter->MaxFilters(); i++){
         
			int filter = m_colorimeter->Filter(i);
			if (filter >= 0) {
				m_filtersCheckedListBox.SetCheck(filter, TRUE);
			}
		}

		for(int i=0; i< m_colorimeter->RangeModeCount(); i++){
			m_rangeModeComboBox.AddString(m_colorimeter->RangeModeName(i));
		}
		m_rangeMode = m_colorimeter->RangeMode();
		m_rangeModeComboBox.SetCurSel(m_rangeMode);

		for(int i=0; i< m_colorimeter->RangeCount(); i++){
			m_rangeComboBox.AddString(m_colorimeter->RangeName(i));
		}
		m_range = m_colorimeter->Range();
		m_rangeComboBox.SetCurSel(m_range);

		for(int i=0; i< m_colorimeter->ExposureModeCount(); i++){
			m_exposureModeComboBox.AddString(m_colorimeter->ExposureModeName(i));
		}
		m_exposureMode = m_colorimeter->ExposureMode();
		m_exposureModeComboBox.SetCurSel(m_exposureMode);
		m_exposure = m_colorimeter->Exposure();;
		m_maxAutoExposure = m_colorimeter->MaxAutoExposure();
		m_minExposure = m_colorimeter->MinExposure();
		m_maxExposure = m_colorimeter->MaxExposure();
	
		m_exposureX = m_colorimeter->ExposureX();
		m_minExposureX = m_colorimeter->MinExposureX();
		m_maxExposureX = m_colorimeter->MaxExposureX();
		m_exposureSpinButton.SetRange(m_colorimeter->MinExposureX(), m_colorimeter->MaxExposureX());

		for(int i=0; i< m_colorimeter->SyncModeCount(); i++){
			m_syncModeComboBox.AddString(m_colorimeter->SyncModeName(i));
		}
		m_syncMode = m_colorimeter->SyncMode();
		m_syncModeComboBox.SetCurSel(m_syncMode);

		m_syncFreq = m_colorimeter->SyncFreq();

		m_minSyncFreq = m_colorimeter->MinSyncFreq();
		m_maxSyncFreq = m_colorimeter->MaxSyncFreq();

		for(int i=0; i< m_colorimeter->UserCalibModeCount(); i++){
			m_userCalibModeComboBox.AddString(m_colorimeter->UserCalibModeName(i));
		}
		m_userCalibMode = m_colorimeter->UserCalibMode();
		m_userCalibModeComboBox.SetCurSel(m_userCalibMode);
	
		for(int i=0; i< m_colorimeter->MatrixCount(m_accessory); i++){
			m_matrixComboBox.AddString(m_colorimeter->MatrixName(m_accessory, i));
		}
		m_matrix = m_colorimeter->Matrix(m_accessory);
		m_matrixComboBox.SetCurSel(m_matrix);
	
		for(int i=0; i< m_colorimeter->Match(); i++){
			m_matchComboBox.AddString(m_colorimeter->MatchName(i));
		}
		m_match = m_colorimeter->Match();
		m_matchComboBox.SetCurSel(m_match);

		
		m_samplingRate = m_colorimeter->SamplingRate();

		m_minSamplingRate = m_colorimeter->MinSamplingRate();
		m_maxSamplingRate = m_colorimeter->MaxSamplingRate();
	
	}
}

void CCR100Setup::UpdateSetup()
{
	m_colorimeter->SetMode(m_mode);

	m_colorimeter->SetAperture(m_aperture);

	m_colorimeter->SetAccessory(m_accessory);

	m_colorimeter->ClearFilters();

	int index = 0;
	for(int filter=0; filter< m_filtersCheckedListBox.GetItemCount(); filter++)
	{
		if(m_filtersCheckedListBox.GetCheck(filter))
		{
			m_colorimeter->SetFilter(index++, filter);
		}
		if(index == m_colorimeter->MaxFilters())
			break;
	}

	m_colorimeter->SetRangeMode(m_rangeMode);
	m_colorimeter->SetRange(m_range);

	m_colorimeter->SetSyncMode(m_syncMode);

	m_colorimeter->SetSyncFreq(m_syncFreq);


	m_colorimeter->SetExposureMode(m_exposureMode);

	m_colorimeter->SetExposure(m_exposure);

	m_colorimeter->SetMaxAutoExposure(m_maxAutoExposure);

	m_colorimeter->SetExposureX(m_exposureX);

	//m_colorimeter->SetMatrixMode(m_matrixMode);
	m_colorimeter->SetUserCalibMode(m_userCalibMode);
	m_colorimeter->SetMatrix(m_accessory, m_matrix);
	m_colorimeter->SetMatch(m_match);
	m_colorimeter->SetSamplingRate(m_samplingRate);
}

void CCR100Setup::OnBnClickedButtonDisconnect()
{
	// TODO: Add your control notification handler code here

	if(m_colorimeter && m_colorimeter->IsConnected())
	{
		m_colorimeter->Disconnect();
	}
	EndDialog(IDCANCEL);

}


void CCR100Setup::OnOK() 
{
	if(UpdateData(TRUE))
	{
		UpdateSetup();
		CDialogEx::OnOK();
	}
}
void CCR100Setup::OnCbnSelchangeComboAccessory()
{
	// TODO: Add your control notification handler code here

	m_matrixComboBox.Clear();
	int accessory = m_accessoryComboBox.GetCurSel();

	for(int i=0; i< m_colorimeter->MatrixCount(accessory); i++){
		m_matrixComboBox.AddString(m_colorimeter->MatrixName(accessory, i));
	}
	m_matrix = m_colorimeter->Matrix(accessory);
	m_matrixComboBox.SetCurSel(m_colorimeter->Matrix(accessory));

}

BOOL CCR100Setup::validateFilters()
{
	int maxFilters = 0;
	for(int filter=0; filter< m_filtersCheckedListBox.GetItemCount(); filter++)
	{
		if(m_filtersCheckedListBox.GetCheck(filter))
		{
			maxFilters++;
		}
	}

    if (maxFilters > m_colorimeter->MaxFilters())
    {
        AfxMessageBox(CStringFormat(_T("A maximum of %d filters can be selected"), m_colorimeter->MaxFilters()));
        return FALSE;
    }
    return TRUE;
}

BOOL CCR100Setup::validateSyncFreq()
{
	if (m_syncFreq < m_minSyncFreq || m_syncFreq > m_maxSyncFreq)
	{
		AfxMessageBox(CStringFormat(_T("Please enter a valid Sync Frequency between %f and %f Hz."), m_minSyncFreq, m_maxSyncFreq));
		return FALSE;
	}
	return TRUE;
}

BOOL CCR100Setup::validateExposure()
{
	if (m_exposure < m_minExposure || m_exposure > m_maxExposure)
	{
		AfxMessageBox(CStringFormat(_T("Please enter a valid Exposure between %f and %f msecs."), m_minExposure, m_maxExposure));
		return FALSE;
	}
	return TRUE;
}


BOOL CCR100Setup::validateMaxAutoExposure()
{
	if (m_maxAutoExposure < m_minExposure || m_maxAutoExposure > m_maxExposure)
	{
		AfxMessageBox(CStringFormat(_T("Please enter a valid Max. Auto Exposure between %f and %f msecs."), m_minExposure, m_maxExposure));
		return FALSE;
	}
	return TRUE;
}
BOOL CCR100Setup::validateExposureX()
{
	return TRUE;
}

BOOL CCR100Setup::validateSamplingRate()
{
	if (m_samplingRate < m_minSamplingRate || m_samplingRate > m_maxSamplingRate)
	{
		AfxMessageBox(CStringFormat(_T("Please enter a valid Sampling Rate between %f and %f Hz."), m_minSamplingRate, m_maxSamplingRate));
		return FALSE;
	}
	return TRUE;
}
