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


// CCR250Setup dialog
class CCRColorimeter;
class CCR250Setup : public CDialogEx
{
	DECLARE_DYNAMIC(CCR250Setup)

public:
	CCR250Setup(CWnd* pParent = NULL);   // standard constructor
	virtual ~CCR250Setup();

	CCRColorimeter* Colorimeter(); 
	void SetColorimeter(CCRColorimeter* colorimeter);

// Dialog Data
	enum { IDD = IDD_CR250SETUP };

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
	
public:
	CButton m_disconnectButton;
	CButton m_okButton;
	CStatic m_versionLabel;
	CComboBox m_apertureComboBox;
	CComboBox m_accessoryComboBox;
	CListCtrl m_filtersCheckedListBox;
	CComboBox m_speedComboBox;
	CComboBox m_exposureModeComboBox;
	CEdit m_exposureTextBox;
	CComboBox m_syncModeComboBox;
	CEdit m_syncFreqTextBox;
	CEdit m_exposureXTextBox;
	int m_mode;
	int m_aperture;
	int m_accessory;
	int m_exposureMode;
	float m_exposure;
	float m_minExposure;
	float m_maxExposure;
	int m_syncMode;
	float m_syncFreq;
	float m_minSyncFreq;
	float m_maxSyncFreq;
	int m_exposureX;
	int m_minExposureX;
	int m_maxExposureX;
	int m_speed;
	CString m_version;
	CSpinButtonCtrl m_exposureSpinButton;
	CComboBox m_colorMatchingFunctionComboBox;
	int m_colorMatchingFunction;
};
