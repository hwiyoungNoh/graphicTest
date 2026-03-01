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
// CR250Setup.cpp : implementation file
//

#include "stdafx.h"
#include "RemoteCommunication.h"
#include "CR250Setup.h"
#include <CRColorimeter.h>


// CCR250Setup dialog

IMPLEMENT_DYNAMIC(CCR250Setup, CDialogEx)

CCR250Setup::CCR250Setup(CWnd* pParent /*=NULL*/)
	: CDialogEx(CCR250Setup::IDD, pParent)
	, m_mode(0)
	, m_aperture(0)
	, m_accessory(0)
	, m_speed(0)
	, m_exposureMode(0)
	, m_exposure(0)
	, m_minExposure(0)
	, m_maxExposure(0)
	, m_syncMode(0)
	, m_syncFreq(0)
	, m_minSyncFreq(0)
	, m_maxSyncFreq(0)
	, m_exposureX(0)
	, m_minExposureX(0)
	, m_maxExposureX(0)
	, m_version(_T(""))
	, m_colorMatchingFunction(0)
{
	m_colorimeter = NULL;
}


CCR250Setup::~CCR250Setup()
{
}

void CCR250Setup::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_BUTTON_DISCONNECT, m_disconnectButton);
	DDX_Control(pDX, IDOK, m_okButton);
	DDX_Control(pDX, IDC_STATIC_VERSION, m_versionLabel);
	DDX_Control(pDX, IDC_COMBO_APERTURE, m_apertureComboBox);
	DDX_Control(pDX, IDC_COMBO_ACCESSORY, m_accessoryComboBox);
	DDX_Control(pDX, IDC_LIST_FILTERS, m_filtersCheckedListBox);
	DDX_Control(pDX, IDC_COMBO_SPEED, m_speedComboBox);
	DDX_Control(pDX, IDC_COMBO_EXPOSUREMODE, m_exposureModeComboBox);
	DDX_Control(pDX, IDC_EDIT_EXPOSURE, m_exposureTextBox);
	DDX_Control(pDX, IDC_COMBO_SYNCMODE, m_syncModeComboBox);
	DDX_Control(pDX, IDC_EDIT_SYNCFREQ, m_syncFreqTextBox);
	DDX_Control(pDX, IDC_EDIT_EXPOSUREX, m_exposureXTextBox);
	DDX_CBIndex(pDX, IDC_COMBO_APERTURE, m_aperture);
	DDX_CBIndex(pDX, IDC_COMBO_ACCESSORY, m_accessory);
	DDX_CBIndex(pDX, IDC_COMBO_SPEED, m_speed);
	DDX_CBIndex(pDX, IDC_COMBO_EXPOSUREMODE, m_exposureMode);
	DDX_Text(pDX, IDC_EDIT_EXPOSURE, m_exposure);
	DDX_CBIndex(pDX, IDC_COMBO_SYNCMODE, m_syncMode);
	DDX_Text(pDX, IDC_EDIT_SYNCFREQ, m_syncFreq);
	DDX_Text(pDX, IDC_EDIT_EXPOSUREX, m_exposureX);
	DDX_Text(pDX, IDC_STATIC_VERSION, m_version);
	DDX_Control(pDX, IDC_SPIN_EXPOSREX, m_exposureSpinButton);;
	DDX_Control(pDX, IDC_COMBO_CMF, m_colorMatchingFunctionComboBox);
	DDX_CBIndex(pDX, IDC_COMBO_CMF, m_colorMatchingFunction);


	//customized data verification
	if(pDX->m_bSaveAndValidate)
	{
		//we will only check data during validation      
		/*if(m_strFilename.IsEmpty())
		{
			CString errorMessage = "The file name field cannot be empty.";
			AfxMessageBox(errorMessage, MB_ICONEXCLAMATION);
			errorMessage.Empty();          

			pDX->PrepareCtrl(IDC_COMBOFILENAME); //make sure pDX will focus to this control
			pDX->Fail();          
		}*/
	}
}


BEGIN_MESSAGE_MAP(CCR250Setup, CDialogEx)
	ON_BN_CLICKED(IDC_BUTTON_DISCONNECT, &CCR250Setup::OnBnClickedButtonDisconnect)
END_MESSAGE_MAP()


// CCR250Setup message handlers
BOOL CCR250Setup::OnInitDialog()
{
	CDialogEx::OnInitDialog();
	// Extra initialization here

	m_filtersCheckedListBox.SetExtendedStyle(m_filtersCheckedListBox.GetStyle()|LVS_EX_CHECKBOXES|LVS_EX_FULLROWSELECT&~LVS_EX_GRIDLINES);
	m_filtersCheckedListBox.InsertColumn(0, _T("Filter"), LVCFMT_LEFT, 150);
	m_filtersCheckedListBox.SetColumnWidth(0, LVSCW_AUTOSIZE_USEHEADER);
	LoadSetup();

	UpdateData(FALSE);

	return TRUE;
}


CCRColorimeter* CCR250Setup::Colorimeter()
{
	return m_colorimeter;
}

void CCR250Setup::SetColorimeter(CCRColorimeter* colorimeter)
{
	m_colorimeter = colorimeter;
}

void CCR250Setup::LoadSetup()
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


		for(int i=0; i< m_colorimeter->SpeedCount(); i++){
			m_speedComboBox.AddString(m_colorimeter->SpeedName(i));
		}
		m_speed = m_colorimeter->Speed();
		m_speedComboBox.SetCurSel(m_speed);

	
		for(int i=0; i< m_colorimeter->ExposureModeCount(); i++){
			m_exposureModeComboBox.AddString(m_colorimeter->ExposureModeName(i));
		}
		m_exposureMode = m_colorimeter->ExposureMode();
		m_exposureModeComboBox.SetCurSel(m_colorimeter->ExposureMode());
		m_exposure = m_colorimeter->Exposure();
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

		m_minSyncFreq = m_colorimeter->MinSyncFreq ();
		m_maxSyncFreq = m_colorimeter->MaxSyncFreq ();

		for(int i=0; i< m_colorimeter->MaxCMF(); i++){
			m_colorMatchingFunctionComboBox.AddString(CStringFormat(_T("User CMF %d"), i+1));
		}
		m_colorMatchingFunctionComboBox.SetCurSel(m_colorMatchingFunction);
	}
}

void CCR250Setup::UpdateSetup()
{
	m_colorimeter->SetAperture(m_aperture);

	m_colorimeter->SetAccessory(m_accessory);

	m_colorimeter->ClearFilters();

	int index = 0;
	for(int filter=0; filter< m_colorimeter->MaxFilters(); filter++)
	{
		if(m_filtersCheckedListBox.GetCheck(filter))
		{
			m_colorimeter->SetFilter(index++, filter);
		}
	}

	m_colorimeter->SetSpeed(m_speed);

	m_colorimeter->SetSyncMode(m_syncMode);

	m_colorimeter->SetSyncFreq(m_syncFreq);


	m_colorimeter->SetExposureMode(m_exposureMode);

	m_colorimeter->SetExposure(m_exposure);

	m_colorimeter->SetExposureX(m_exposureX);
	
	m_colorimeter->SetCMF(m_colorMatchingFunction);


}

void CCR250Setup::OnBnClickedButtonDisconnect()
{
	// TODO: Add your control notification handler code here

	if(m_colorimeter && m_colorimeter->IsConnected())
	{
		m_colorimeter->Disconnect();
	}
	EndDialog(IDCANCEL);

}


void CCR250Setup::OnOK() 
{
	if(UpdateData(TRUE))
	{
		UpdateSetup();
		CDialogEx::OnOK();
	}
}

