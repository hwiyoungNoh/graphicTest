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
#include "afxwin.h"
#include "afxcmn.h"


// CCR100Setup dialog
class CCRColorimeter;

class CCR100Setup : public CDialogEx
{
	DECLARE_DYNAMIC(CCR100Setup)

public:
	CCR100Setup(CWnd* pParent = NULL);   // standard constructor
	virtual ~CCR100Setup();

	CCRColorimeter* Colorimeter(); 
	void SetColorimeter(CCRColorimeter* colorimeter);

// Dialog Data
	enum { IDD = IDD_CR100SETUP };

protected:
	CCRColorimeter* m_colorimeter;
	void UpdateSetup();
	void LoadSetup();
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	DECLARE_MESSAGE_MAP()
	BOOL OnInitDialog();
	void OnOK();
	afx_msg void OnBnClickedButtonDisconnect();

private:
	BOOL validateFilters();
	BOOL validateSyncFreq();
	BOOL validateExposure();
	BOOL validateMaxAutoExposure();
	BOOL validateExposureX();
	BOOL validateSamplingRate();

public:
	CButton m_disconnectButton;
	CButton m_okButton;
	CStatic m_versionLabel;
	CComboBox m_modeComboBox;
	CComboBox m_apertureComboBox;
	CComboBox m_accessoryComboBox;
	CListCtrl m_filtersCheckedListBox;
	CComboBox m_rangeModeComboBox;
	CComboBox m_rangeComboBox;
	CComboBox m_exposureModeComboBox;
	CEdit m_exposureTextBox;
	CEdit m_maxAutoExposureTextBox;
	CComboBox m_syncModeComboBox;
	CEdit m_syncFreqTextBox;
	CEdit m_exposureXTextBox;
	CComboBox m_userCalibModeComboBox;
	CComboBox m_matrixComboBox;
	CComboBox m_matchComboBox;
	int m_mode;
	int m_aperture;
	int m_accessory;
	int m_rangeMode;
	int m_exposureMode;
	float m_exposure;
	float m_maxAutoExposure;
	float m_minExposure;
	float m_maxExposure;
	int m_syncMode;
	float m_syncFreq;
	float m_minSyncFreq;
	float m_maxSyncFreq;
	int m_exposureX;
	int m_minExposureX;
	int m_maxExposureX;
	int m_range;
	int m_userCalibMode;
	int m_matrix;
	int m_match;
	CString m_version;
	CSpinButtonCtrl m_exposureSpinButton;
	CSpinButtonCtrl m_maxAutoExposureSpinButton;
	CButton m_matrixEditButton;
	CButton m_matchEditButton;
	afx_msg void OnCbnSelchangeComboAccessory();
	CEdit m_samplingRateTextBox;
	float m_samplingRate;
	float m_minSamplingRate;
	float m_maxSamplingRate;
};
